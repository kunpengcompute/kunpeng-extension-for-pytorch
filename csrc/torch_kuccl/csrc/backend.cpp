/*
 * KUCCL Backend 实现 —— c10d::BackendKuccl
 *
 * 类声明在 include/kuccl/backend.hpp；PYBIND11_MODULE 在文件末尾；
 * __attribute__((constructor)) 自动注册（import 即注册）。
 *
 * 底层委托（异步 C API + kucclTest 轮询）：
 *   构造       → kucclGetUniqueId + kuccl::oob_allgather 广播 + kucclCommInitRankConfig
 *   allgather  → kucclAllGather
 *   allreduce  → kucclAllReduce
 *   broadcast  → kucclBroadcast
 *   alltoall   → kucclAllToAll（等长直调；变长 split 不支持）
 *   send/recv  → 不支持（只有 4 个集合接口，见 kuccl.h）
 *   barrier    → 1 字节 allreduce / KUCCL_BARRIER=store 回退
 *
 * C API 是异步的，Backend 期望阻塞：启动 request 后用 kucclTest 循环等完成。
 */

#include "backend.hpp" // 本包头文件（适配层，torch 依赖在这）

#include <torch/csrc/Exceptions.h> // TORCH_WARN（后端自动回退提示）

#include <kuccl.h> // libkuccl 的对外 C 接口（零 torch）

#include <algorithm> // C++ 标准库
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "oob.hpp" // Store 版 OOB 工具（适配层私有）

namespace c10d {

namespace {

/* allreduce 默认 clone 一份 sendbuf（输入独立快照，正确性与底层算法
 * 实现解耦）。KUCCL_ALLREDUCE_INPLACE=1 时原地（sendbuff==recvbuff）。 */
bool allreduce_inplace_enabled() {
    static const bool v = [] {
        const char *e = std::getenv("KUCCL_ALLREDUCE_INPLACE");
        return e != nullptr && e[0] != '\0' && std::strcmp(e, "0") != 0;
    }();
    return v;
}

/* ── allreduce 输入快照：独立副本，正确性与底层算法解耦 ──────────
 *
 * 拷贝路径按消息大小分段（默认分界 4MB，可调）：
 *   ≤ TORCH_KUCCL_SNAPSHOT_SERIAL_MAX：串行 memcpy（不修改线程配置）
 *   更大：线程数收敛到可用核数后 clone，结束恢复
 *
 * 真实核数 = 调用线程绑定的 CPU 集合（sched_getaffinity）∩ cgroup CPU
 * 配额。绑定集合是线程数上限（超过必然超订），但可能与他人共享。
 * 收敛只缩不放：宿主线程数不超过该值时不动，尊重上层配置；
 * RAII 保证异常路径恢复。 */

/* 分段阈值（字节）：TORCH_KUCCL_SNAPSHOT_SERIAL_MAX 可按部署环境重标 */
static size_t package_threshold_mt() {
    static const size_t v = [] {
        if (const char *e = std::getenv("TORCH_KUCCL_SNAPSHOT_SERIAL_MAX")) {
            const unsigned long long b = std::strtoull(e, nullptr, 10);
            if (b > 0)
                return static_cast<size_t>(b);
        }
        return static_cast<size_t>(4) << 20; /* 4MB */
    }();
    return v;
}

/* cgroup CPU 配额折算核数（>0 为 N 核，-1 无配额）；进程内缓存 */
static int cpu_quota_cpus() {
    static const int q = [] {
        char v[32];
        unsigned long long period = 0;
        /* cgroup v2: "<quota> <period>"，"max" = 不限 */
        if (FILE *f = std::fopen("/sys/fs/cgroup/cpu.max", "r")) {
            const int n = std::fscanf(f, "%31s %llu", v, &period);
            std::fclose(f);
            if (n == 2 && std::strcmp(v, "max") != 0 && period > 0) {
                const unsigned long long c = (std::strtoull(v, nullptr, 10) + period - 1) / period;
                return c > 1000000 ? -1 : static_cast<int>(c);
            }
            return -1;
        }
        /* cgroup v1: quota 与 period 分文件（quota <= 0 = 不限） */
        long long quota = -1;
        if (FILE *f = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {
            if (std::fscanf(f, "%lld", &quota) != 1)
                quota = -1;
            std::fclose(f);
        }
        if (quota <= 0)
            return -1;
        long long p = 100000;
        if (FILE *f = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) {
            if (std::fscanf(f, "%lld", &p) != 1 || p <= 0)
                p = 100000;
            std::fclose(f);
        }
        const long long c = (quota + p - 1) / p;
        return c > 1000000 ? -1 : static_cast<int>(c);
    }();
    return q;
}

/* 调用线程被绑定（允许运行）的 CPU 个数；探测失败返回 0。
 * 绑定集合是线程数上限而非独占核数（可能与其他进程共享）。 */
static int bound_cpu_count() {
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) != 0)
        return 0;
    return std::max(CPU_COUNT(&set), 0);
}

/* clone 期间收敛到的线程数上限：绑定 CPU 集合 ∩ cgroup 配额；
 * 0 = 探测失败（不干预） */
static int clone_thread_cap() {
    const int bound = bound_cpu_count();
    if (bound <= 0)
        return 0;
    const int quota = cpu_quota_cpus();
    return quota <= 0 ? bound : std::max(1, std::min(bound, quota));
}

/* RAII：clone 期间收敛线程数，析构恢复（异常路径同样恢复）；
 * 只缩不放，宿主线程数本就 ≤ cap 时零干预 */
class ScopedCloneThreads {
public:
    ScopedCloneThreads() {
        const int cap = clone_thread_cap();
        if (cap <= 0)
            return;
        const int cur = at::get_num_threads();
        if (cap >= cur)
            return;
        old_ = cur;
        at::set_num_threads(cap);
    }
    ~ScopedCloneThreads() {
        if (old_ > 0)
            at::set_num_threads(old_);
    }
    ScopedCloneThreads(const ScopedCloneThreads &) = delete;
    ScopedCloneThreads &operator=(const ScopedCloneThreads &) = delete;

private:
    int old_ = -1;
};

/* 串行拷贝（不修改线程配置） */
static at::Tensor memcpy_st(const at::Tensor &t) {
    at::Tensor out = at::empty_like(t);
    const size_t bytes =
        static_cast<size_t>(t.numel()) * static_cast<size_t>(t.element_size());
    if (bytes > 0)
        std::memcpy(out.data_ptr(), t.data_ptr(), bytes);
    return out;
}

/* 并行拷贝：收敛到真实核数后 clone，完事恢复 */
static at::Tensor memcpy_mt(const at::Tensor &t) {
    if (at::in_parallel_region()) /* 并行区内 clone 自动串行，且不应改线程数 */
        return t.clone();
    const ScopedCloneThreads cap;
    return t.clone();
}

/* 拷贝主入口：按消息大小自适应选拷贝策略（见顶部说明） */
at::Tensor adaptive_memcpy(const at::Tensor &t) {
    const size_t bytes =
        static_cast<size_t>(t.numel()) * static_cast<size_t>(t.element_size());
    if (bytes <= package_threshold_mt())
        return memcpy_st(t);
    return memcpy_mt(t);
}

/* 进程内一次的诊断输出：分段阈值 + 收敛上限 + 宿主池超配预警 */
static void log_clone_census() {
    static const bool once = [] {
        const int bound = bound_cpu_count();
        const int quota = cpu_quota_cpus();
        const int real = (bound <= 0 || quota <= 0) ? bound : std::max(1, std::min(bound, quota));
        const int host = at::get_num_threads();
        std::fprintf(stderr,
                     "[KuCCL] memcpy: threshold_mt=%zuB cap=%d (bound=%d, quota=%d, "
                     "real=%d; host torch pool=%d)\n",
                     package_threshold_mt(), clone_thread_cap(), bound, quota, real, host);
        if (real > 0 && host > real * 2)
            std::fprintf(stderr,
                         "[KuCCL] WARNING: host thread pool %d >> real cpus %d; "
                         "parallel_for barriers will be amplified, set "
                         "OMP_NUM_THREADS=%d before launch\n",
                         host, real, real);
        return true;
    }();
    (void)once;
}


/* barrier 走 1 字节 allreduce；KUCCL_BARRIER=store 回退 TCPStore 版。 */
bool barrier_use_store() {
    static const bool v = [] {
        const char *e = std::getenv("KUCCL_BARRIER");
        return e != nullptr && std::strcmp(e, "store") == 0;
    }();
    return v;
}

} // namespace

namespace {
/* 异步 Work 构造辅助 */
WorkKucclPtr make_async_work(kucclRequest_t req, kucclComm_t comm, std::vector<at::Tensor> tensors,
                             c10d::OpType op_type) {
    return c10::make_intrusive<WorkKuccl>(req, comm, std::move(tensors), op_type);
}
} // namespace

/* ── dtype / reduceop 转换（静态成员，声明在 backend.hpp）────────── */
kucclDataType_t BackendKuccl::torchToKucclDtype(c10::ScalarType st) {
    switch (st) {
    case c10::kChar:
        return kucclInt8;
    case c10::kByte:
        return kucclUint8;
    case c10::kInt:
        return kucclInt32;
    case c10::kLong:
        return kucclInt64;
    case c10::kFloat:
        return kucclFloat;
    case c10::kDouble:
        return kucclDouble;
    default:
        TORCH_CHECK(false, "kuccl unsupported dtype: ", st);
    }
}

kucclRedOp_t BackendKuccl::torchToKucclRedOp(const ReduceOp &op) {
    if (op == ReduceOp::SUM)
        return kucclSum;
    if (op == ReduceOp::MIN)
        return kucclMin;
    if (op == ReduceOp::MAX)
        return kucclMax;
    TORCH_CHECK(false, "kuccl only supports SUM/MIN/MAX reduce op");
}

/* ════════════════════════════════════════════════════════════
 * PyTorch Store → kuccl 库 OOB 回调的桥接
 *
 * kuccl 纯库（libkuccl.so）不认识 c10d::Store（OOB 留给调用方注入）。
 * 适配层在这里把 Store 封装成 config.oob_allgather 需要的函数指针。
 *
 * 回调签名（types.hpp）：int(const void* send, void* recv, int count, void* ctx)
 * 语义对齐 MPI_Allgather：每 rank 贡献 count 字节，收齐后按 rank 拼接。
 * StoreOobCtx 完整定义在 backend.hpp（public 嵌套结构）。
 * ════════════════════════════════════════════════════════════*/
namespace {

/* Store 版 allgather 回调：把 kuccl 库的 OOB 请求转成 kuccl::oob_allgather */
int store_oob_allgather_cb(const void *sendbuf, void *recvbuf, int count, void *user_ctx) {
    auto *ctx = static_cast<BackendKuccl::StoreOobCtx *>(user_ctx);
    if (!ctx || !ctx->store)
        return -1;
    /* 轮次递增 → key 前缀唯一，避免多次 OOB 交换互相覆盖 */
    const std::uint64_t r = ctx->round.fetch_add(1, std::memory_order_relaxed);
    const std::string prefix = "kuccl_oob_" + std::to_string(r) + "_";
    try {
        kuccl::oob_allgather(ctx->store, sendbuf, recvbuf, ctx->rank_hint, ctx->ws_hint,
                             static_cast<std::size_t>(count), prefix);
        return 0;
    } catch (const std::exception &) {
        return -1;
    }
}

} /* anonymous namespace */

/* ── 构造函数 ──────────────────────────────────────────────
 *
 * 适配层职责（构造期）：
 *   1. 用 Store 把 rank 0 的 uniqueId 广播给所有 rank（rendezvous）
 *   2. 构造 kucclConfig_t：注入 Store 版 OOB 回调（kuccl 库据此完成引导）
 *   3. kucclCommInitRankConfig 创建通信域（进程级单例 + 两阶段地址交换在库内完成）
 *
 * store_ 是 PyTorch 传入的 per-group PrefixStore（组隔离天然成立）。 */
BackendKuccl::BackendKuccl(c10::intrusive_ptr<Store> store, int rank, int world_size,
                           std::chrono::milliseconds /*timeout*/)
    : Backend(rank, world_size), store_(std::move(store)) {
    /* 拷贝分段诊断（进程内一次：真实核数 + 宿主池超配预警） */
    log_clone_census();

    /* Store OOB 上下文（含 ws/rank 提示，回调时用）—— 直接成员，随 this 存活 */
    store_oob_.store = store_.get();
    store_oob_.rank_hint = rank;
    store_oob_.ws_hint = world_size;

    /* 1. rank 0 生成 uniqueId，通过 Store 广播（所有 rank 必须拿到同一个 ID） */
    kucclUniqueId uniqueId;
    std::memset(&uniqueId, 0, sizeof(uniqueId));
    {
        const std::uint64_t rnd = store_oob_.round.fetch_add(1, std::memory_order_relaxed);
        const std::string call_prefix = "kuccl_uid_" + std::to_string(rnd) + "_";
        if (rank == 0) {
            kucclResult_t r = kucclGetUniqueId(&uniqueId);
            TORCH_CHECK(r == KUCCL_SUCCESS, "kucclGetUniqueId failed (result=", (int)r, ")");
        }
        std::vector<char> id_recv(static_cast<size_t>(world_size) * sizeof(uniqueId), 0);
        kuccl::oob_allgather(store_oob_.store, &uniqueId, id_recv.data(), rank, world_size,
                             sizeof(uniqueId), call_prefix);
        std::memcpy(uniqueId.data, id_recv.data(), sizeof(uniqueId)); /* 取 rank 0 的 */
    }

    /* 2. 构造 config：注入 Store 版 OOB 回调（kuccl 库不认识 Store） */
    kucclConfig_t config;
    std::memset(&config, 0, sizeof(config));
    config.oob_allgather = &store_oob_allgather_cb;
    config.oob_bcast = nullptr; /* 广播用 allgather 语义覆盖 */
    config.oob_ctx = static_cast<void *>(&store_oob_);

    /* 3. 后端自动选择：单机（LOCAL_WORLD_SIZE == WORLD_SIZE，torchrun 注入）
     *    自动走 kupl shm 快路径；用户显式设了 KUCCL_BACKEND_TYPE 则完全尊重。
     *    KUPL 未编入/不可用时自动回退 xucg（见下方重试）。 */
    bool auto_kupl = false;
    {
        const char *pinned = std::getenv("KUCCL_BACKEND_TYPE");
        if (pinned == nullptr || pinned[0] == '\0') {
            const char *local = std::getenv("LOCAL_WORLD_SIZE");
            const char *world = std::getenv("WORLD_SIZE");
            if (local != nullptr && world != nullptr && std::strcmp(local, world) == 0) {
                config.backend = "kupl";
                auto_kupl = true;
            }
        }
    }

    /* 4. 创建通信域（自动选中的 kupl 不可用时回退 xucg 重建） */
    kucclResult_t r = kucclCommInitRankConfig(&comm_, world_size, uniqueId, rank, &config);
    if (r != KUCCL_SUCCESS && auto_kupl) {
        TORCH_WARN("kuccl: kupl backend unavailable (not built in or init failed), "
                   "falling back to xucg");
        config.backend = "xucg";
        r = kucclCommInitRankConfig(&comm_, world_size, uniqueId, rank, &config);
    }
    TORCH_CHECK(r == KUCCL_SUCCESS, "kucclCommInitRankConfig failed (result=", (int)r, ")");
    comm_ready_ = true;
}

BackendKuccl::~BackendKuccl() {
    if (comm_ready_) {
        kucclCommDestroy(comm_);
        comm_ = nullptr;
        comm_ready_ = false;
    }
}

/* ── shutdown：c10d destroy_process_group 的调用点。
 * 先 abort，使在飞 Work 后续 poll 得到 ERROR 而非触碰已销毁的 group。 */
void BackendKuccl::shutdown() {
    if (comm_ready_ && comm_) {
        kucclCommAbort(comm_);
    }
}

/* ── allgather（列表式）────────────────────────────────── */
c10::intrusive_ptr<Work>
BackendKuccl::allgather(std::vector<std::vector<at::Tensor>> &outputTensors,
                        std::vector<at::Tensor> &inputTensors, const AllgatherOptions &) {
    TORCH_CHECK(inputTensors.size() == 1, "kuccl allgather expects exactly 1 input tensor");
    TORCH_CHECK(outputTensors.size() == 1, "kuccl allgather expects exactly 1 output list");

    const int ws = getSize();
    auto &outList = outputTensors[0];
    TORCH_CHECK(static_cast<int>(outList.size()) == ws, "output list size != world_size");

    at::Tensor input = inputTensors[0].contiguous();
    TORCH_CHECK(input.is_cpu(), "kuccl backend only supports CPU tensors");

    const int64_t count = input.numel();
    const kucclDataType_t dt = torchToKucclDtype(input.scalar_type());
    const int64_t total = static_cast<int64_t>(ws) * count;

    /* 中转缓冲复用：dtype 一致且容量足够则复用，否则重建 */
    at::Tensor recv;
    if (allgather_scratch_.defined() && allgather_scratch_.scalar_type() == input.scalar_type() &&
        allgather_scratch_.numel() >= total) {
        recv = allgather_scratch_.narrow(0, 0, total);
    } else {
        allgather_scratch_ = at::empty({total}, input.options());
        recv = allgather_scratch_;
    }

    kucclRequest_t req = nullptr;
    kucclResult_t r = kucclAllGather(input.data_ptr(), recv.data_ptr(), static_cast<size_t>(count),
                                     dt, comm_, &req);
    TORCH_CHECK(r == KUCCL_SUCCESS, "kucclAllGather start failed (result=", (int)r, ")");
    /* 真异步：立即返回 Work；拆回拷贝作为 finisher（完成时执行） */
    auto finisher = [ws, count, recv, outList]() {
        for (int i = 0; i < ws; ++i) {
            outList[i].copy_(recv.narrow(0, i * count, count).view_as(outList[i]));
        }
    };
    return c10::make_intrusive<WorkKuccl>(req, comm_, std::vector<at::Tensor>{}, OpType::ALLGATHER,
                                          std::move(finisher));
}

/* ── _allgather_base（all_gather_into_tensor）────────────── */
c10::intrusive_ptr<Work> BackendKuccl::_allgather_base(at::Tensor &outputBuffer,
                                                       at::Tensor &inputBuffer,
                                                       const AllgatherOptions &) {
    TORCH_CHECK(comm_ready_, "kuccl _allgather_base: comm is not initialized");
    TORCH_CHECK(inputBuffer.defined() && outputBuffer.defined(), "undefined tensor");
    TORCH_CHECK(inputBuffer.device().is_cpu() && outputBuffer.device().is_cpu(),
                "kuccl _allgather_base currently only supports CPU tensor");
    TORCH_CHECK(outputBuffer.is_contiguous(), "requires contiguous output tensor");
    TORCH_CHECK(inputBuffer.scalar_type() == outputBuffer.scalar_type(),
                "requires the same dtype for input and output");

    const int64_t world_size = static_cast<int64_t>(getSize());
    at::Tensor input = inputBuffer.contiguous();
    const int64_t input_numel = input.numel();

    TORCH_CHECK(outputBuffer.numel() == input_numel * world_size, "output size mismatch");

    if (input_numel == 0) {
        return c10::make_intrusive<WorkKuccl>(OpType::ALLGATHER);
    }

    const size_t count = static_cast<size_t>(input_numel);
    const kucclDataType_t dt = torchToKucclDtype(input.scalar_type());

    TORCH_CHECK(count <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                "per-rank element count too large for int32 range");

    kucclRequest_t req = nullptr;
    kucclResult_t r =
        kucclAllGather(input.data_ptr(), outputBuffer.data_ptr(), count, dt, comm_, &req);
    TORCH_CHECK(r == KUCCL_SUCCESS, "kuccl _allgather_base start failed: rank=", getRank(),
                ", count=", count, ", world_size=", world_size);
    /* 异步保活：input 可能是 contiguous() 的临时副本，outputBuffer 是写目标 */
    return make_async_work(req, comm_, {input, outputBuffer}, OpType::ALLGATHER);
}

/* ── broadcast ──────────────────────────────────────────── */
c10::intrusive_ptr<Work> BackendKuccl::broadcast(std::vector<at::Tensor> &tensors,
                                                 const BroadcastOptions &opts) {
    TORCH_CHECK(tensors.size() == 1, "kuccl broadcast 仅支持每次 1 个 tensor");
    at::Tensor &t = tensors[0];
    TORCH_CHECK(t.is_contiguous(), "kuccl broadcast 需要 contiguous tensor");
    TORCH_CHECK(t.device().is_cpu(), "kuccl broadcast 目前仅支持 CPU tensor");

    const int root = static_cast<int>(opts.rootRank);
    const size_t count = static_cast<size_t>(t.numel());
    const kucclDataType_t dt = torchToKucclDtype(t.scalar_type());

    /* PyTorch broadcast 是原地操作：root 的数据既是源也是目的，sendbuff 与 recvbuff 同址 */
    kucclRequest_t req = nullptr;
    kucclResult_t r = kucclBroadcast(t.data_ptr(), t.data_ptr(), count, dt, root, comm_, &req);
    TORCH_CHECK(r == KUCCL_SUCCESS, "kucclBroadcast start 失败 (result=", (int)r, ")");
    /* 异步保活：原地操作，UCG 在飞期间读写 t */
    return make_async_work(req, comm_, {t}, OpType::BROADCAST);
}

/* ── allreduce ────────────────────────────────────────────
 *
 * 实现 coalesced（合并归约）策略：多 tensor 拼成连续 buffer，
 * 一轮通信完成（同 DDP Reducer 的 bucket flatten 思路）。
 *
 * 单 tensor（tensors.size()==1）：直接调 kucclAllReduce，零开销。
 * 多 tensor（tensors.size()>1）：拼成连续 buffer → 1 次 allreduce → 拆回。
 *   比 for 循环快 N 倍（for 循环做 N 轮通信，coalesced 只做 1 轮）。
 *
 * 约束：所有 tensor 必须 dtype 相同、CPU、contiguous。
 * 这是 PyTorch allreduce_coalesced 的标准约束。 */
c10::intrusive_ptr<Work> BackendKuccl::allreduce(std::vector<at::Tensor> &tensors,
                                                 const AllreduceOptions &opts) {
    TORCH_CHECK(!tensors.empty(), "kuccl allreduce: empty tensor list");

    /* 单 tensor 快速路径：直接调，不拼 buffer */
    if (tensors.size() == 1) {
        at::Tensor &t = tensors[0];
        TORCH_CHECK(t.is_contiguous(), "kuccl allreduce needs contiguous tensor");
        TORCH_CHECK(t.device().is_cpu(), "kuccl allreduce only supports CPU tensor");

        const size_t count = static_cast<size_t>(t.numel());
        const kucclDataType_t dt = torchToKucclDtype(t.scalar_type());
        const kucclRedOp_t op = torchToKucclRedOp(opts.reduceOp);

        /* 原地归约（sendbuff==recvbuff）免全量拷贝 */
        at::Tensor sendbuf; /* clone 路径才持有副本 */
        void *send_ptr = t.data_ptr();
        if (!allreduce_inplace_enabled()) {
            sendbuf = adaptive_memcpy(t);
            send_ptr = sendbuf.data_ptr();
        }
        kucclRequest_t req = nullptr;
        kucclResult_t r = kucclAllReduce(send_ptr, t.data_ptr(), count, dt, op, comm_, &req);
        TORCH_CHECK(r == KUCCL_SUCCESS, "kucclAllReduce start failed (result=", (int)r, ")");
        /* 异步保活：t 是写目标；sendbuf（clone 路径）是读源——
         * 离开作用域后 UCG 仍在访问其内存，必须随 Work 存活 */
        std::vector<at::Tensor> keep;
        keep.push_back(t);
        if (sendbuf.defined())
            keep.push_back(sendbuf);
        return make_async_work(req, comm_, std::move(keep), OpType::ALLREDUCE);
    }

    /* 多 tensor coalesced 路径：拼 buffer → 1 次 allreduce → 拆回 */
    const c10::ScalarType dtype = tensors[0].scalar_type();
    const size_t elem_size = static_cast<size_t>(tensors[0].element_size());
    const kucclDataType_t dt = torchToKucclDtype(dtype);
    const kucclRedOp_t op = torchToKucclRedOp(opts.reduceOp);

    /* 校验所有 tensor：同 dtype、CPU、contiguous */
    int64_t total_numel = 0;
    std::vector<int64_t> offsets;
    offsets.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        TORCH_CHECK(tensors[i].scalar_type() == dtype,
                    "kuccl allreduce_coalesced: all tensors must have the same dtype");
        TORCH_CHECK(tensors[i].device().is_cpu(), "kuccl allreduce only supports CPU tensor");
        at::Tensor c = tensors[i].contiguous();
        tensors[i] = c; /* 原地替换为 contiguous 版本 */
        offsets.push_back(total_numel);
        total_numel += tensors[i].numel();

        TORCH_CHECK(total_numel <= std::numeric_limits<int32_t>::max(),
                    "kuccl allreduce_coalesced: total numel exceeds int32 range");
    }

    /* 拼成一个大 buffer（flat_ 不拷贝连续 tensor） */
    at::Tensor flat = at::empty({total_numel}, tensors[0].options());

    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].numel() > 0) {
            std::memcpy(static_cast<char *>(flat.data_ptr()) + offsets[i] * elem_size,
                        tensors[i].data_ptr(), static_cast<size_t>(tensors[i].numel()) * elem_size);
        }
    }

    /* 1 次 allreduce 通信 */
    at::Tensor sendbuf; /* clone 路径才持有副本 */
    void *send_ptr = flat.data_ptr();
    if (!allreduce_inplace_enabled()) {
        sendbuf = adaptive_memcpy(flat);
        send_ptr = sendbuf.data_ptr();
    }

    kucclRequest_t req = nullptr;
    kucclResult_t r = kucclAllReduce(send_ptr, flat.data_ptr(), static_cast<size_t>(total_numel),
                                     dt, op, comm_, &req);
    TORCH_CHECK(r == KUCCL_SUCCESS, "kuccl allreduce_coalesced start failed (result=", (int)r, ")");
    /* 真异步：拆回拷贝作为 finisher；sendbuf/flat 由 tensors_ 保活 */
    auto finisher = [tensors, flat, offsets, elem_size]() {
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (tensors[i].numel() > 0) {
                std::memcpy(tensors[i].data_ptr(),
                            static_cast<const char *>(flat.data_ptr()) + offsets[i] * elem_size,
                            static_cast<size_t>(tensors[i].numel()) * elem_size);
            }
        }
    };
    std::vector<at::Tensor> keep;
    keep.reserve(tensors.size() + 2);
    for (const auto &t : tensors)
        keep.push_back(t);
    keep.push_back(flat);
    if (sendbuf.defined())
        keep.push_back(sendbuf);
    return c10::make_intrusive<WorkKuccl>(req, comm_, std::move(keep), OpType::ALLREDUCE,
                                          std::move(finisher));
}

/* ── allreduce_coalesced ─────────────────────────────────
 *
 * PyTorch 标准的 allreduce_coalesced 接口，直接委托给上面的 allreduce。
 * 从 PyTorch 2.x 起这个接口的参数就是 (tensors, opts)，和 allreduce 一样。 */
c10::intrusive_ptr<Work> BackendKuccl::allreduce_coalesced(std::vector<at::Tensor> &tensors,
                                                           const AllreduceCoalescedOptions &opts) {
    AllreduceOptions ar_opts;
    ar_opts.reduceOp = opts.reduceOp;
    return allreduce(tensors, ar_opts);
}

/* ── alltoall_base ────────────────────────────────────────
 *
 * V2 的 kucclAllToAll 只支持等长交换（每对进程 count 个元素）。
 * 等长 split：直接调 kucclAllToAll（零拷贝）。
 * 变长 split：把每对进程的数据 padding 到最大 slot 大小，拼成等长后
 *             一次 kucclAllToAll，再从 recv slot 拆回。
 * 全部以 kucclUint8 做字节级搬运（与旧 alltoallv 的 UINT8 策略一致）。 */
c10::intrusive_ptr<Work> BackendKuccl::alltoall_base(at::Tensor &outputBuffer,
                                                     at::Tensor &inputBuffer,
                                                     std::vector<int64_t> &outputSplitSizes,
                                                     std::vector<int64_t> &inputSplitSizes,
                                                     const AllToAllOptions &) {
    TORCH_CHECK(comm_ready_, "kuccl alltoall_base: comm is not initialized");
    TORCH_CHECK(inputBuffer.defined() && outputBuffer.defined(), "undefined tensor");
    TORCH_CHECK(inputBuffer.device().is_cpu() && outputBuffer.device().is_cpu(),
                "kuccl alltoall_base only supports CPU tensors");
    TORCH_CHECK(inputBuffer.scalar_type() == outputBuffer.scalar_type(),
                "kuccl alltoall_base requires same dtype");
    TORCH_CHECK(inputBuffer.dim() >= 1 && outputBuffer.dim() >= 1,
                "kuccl alltoall_base requires tensor dim >= 1");
    TORCH_CHECK(inputBuffer.dim() == outputBuffer.dim(), "input/output dim mismatch");
    TORCH_CHECK(outputBuffer.is_contiguous(), "requires contiguous output");

    at::Tensor input = inputBuffer.contiguous();
    for (int64_t d = 1; d < input.dim(); ++d) {
        TORCH_CHECK(input.size(d) == outputBuffer.size(d), "trailing shape mismatch at dim ", d);
    }

    const int ws = getSize();

    auto normalize_splits = [ws](const at::Tensor &tensor, const std::vector<int64_t> &splits,
                                 const char *name) {
        if (splits.empty()) {
            TORCH_CHECK(tensor.size(0) % ws == 0, name, ": tensor.size(0)=", tensor.size(0),
                        " not divisible by world_size=", ws);
            return std::vector<int64_t>(static_cast<size_t>(ws), tensor.size(0) / ws);
        }
        TORCH_CHECK(splits.size() == static_cast<size_t>(ws), name, " size=", splits.size(),
                    ", expected=", ws);
        int64_t total = 0;
        for (int i = 0; i < ws; ++i) {
            TORCH_CHECK(splits[i] >= 0, name, "[", i, "] is negative");
            total += splits[i];
        }
        TORCH_CHECK(total == tensor.size(0), name, " sum=", total,
                    ", tensor.size(0)=", tensor.size(0));
        return splits;
    };

    const auto input_splits = normalize_splits(input, inputSplitSizes, "inputSplitSizes");
    const auto output_splits = normalize_splits(outputBuffer, outputSplitSizes, "outputSplitSizes");

    int64_t row_elems = 1;
    for (int64_t d = 1; d < input.dim(); ++d)
        row_elems *= input.size(d);

    const int64_t elem_size = static_cast<int64_t>(input.element_size());
    const int64_t row_bytes = row_elems * elem_size;

    /* 每对进程的字节数与位移 */
    std::vector<int64_t> send_bytes(ws, 0), sdispls(ws, 0);
    std::vector<int64_t> recv_bytes(ws, 0), rdispls(ws, 0);
    int64_t send_offset = 0, recv_offset = 0;

    for (int i = 0; i < ws; ++i) {
        const int64_t sb = input_splits[i] * row_bytes;
        const int64_t rb = output_splits[i] * row_bytes;
        TORCH_CHECK(sb <= INT32_MAX && rb <= INT32_MAX && send_offset <= INT32_MAX &&
                        recv_offset <= INT32_MAX,
                    "exceeds UCG int32 range");
        send_bytes[i] = sb;
        sdispls[i] = send_offset;
        recv_bytes[i] = rb;
        rdispls[i] = recv_offset;
        send_offset += sb;
        recv_offset += rb;
    }

    /* 判定等长：所有 send_bytes 相同且所有 recv_bytes 相同且两者相等 */
    bool equal_split = true;
    for (int i = 1; i < ws; ++i) {
        if (send_bytes[i] != send_bytes[0] || recv_bytes[i] != recv_bytes[0]) {
            equal_split = false;
            break;
        }
    }
    equal_split = equal_split && (send_bytes[0] == recv_bytes[0]);

    /* ── 等长快速路径：直接 kucclAllToAll，零拷贝 ── */
    if (equal_split) {
        const size_t per_rank = static_cast<size_t>(send_bytes[0]);
        if (per_rank == 0) {
            return c10::make_intrusive<WorkKuccl>(OpType::ALLTOALL);
        }

        kucclRequest_t req = nullptr;
        kucclResult_t r = kucclAllToAll(input.data_ptr(), outputBuffer.data_ptr(), per_rank,
                                        kucclUint8, comm_, &req);
        TORCH_CHECK(r == KUCCL_SUCCESS, "kucclAllToAll start failed: rank=", getRank(),
                    " (result=", (int)r, ")");
        /* 异步保活：input（可能 contiguous 临时）+ outputBuffer */
        return make_async_work(req, comm_, {input, outputBuffer}, OpType::ALLTOALL);
    }

    /* ── 变长 split：不支持（kuccl 公开面只保留等长 AllToAll）── */
    TORCH_CHECK(false, "kuccl alltoall with variable split sizes is not supported: ",
                "the public API is scoped to equal-size AllToAll. ",
                "Use equal splits, or the gloo backend for variable alltoall.");
}

/* ── send / recv（点对点）──────────────────────────────────
 * kuccl 公开面收敛为 4 个集合通信接口（AllGather/AllReduce/Broadcast/
 * AllToAll，见 kuccl.h），点对点不在承诺范围内——显式报错而非静默。 */
c10::intrusive_ptr<Work> BackendKuccl::send(std::vector<at::Tensor> & /*tensors*/, int /*dstRank*/,
                                            int /*tag*/) {
    TORCH_CHECK(false, "kuccl backend does not support send/recv: the public API is ",
                "scoped to 4 collectives (allgather/allreduce/broadcast/alltoall). ",
                "Use gloo/mpi for point-to-point.");
}

c10::intrusive_ptr<Work> BackendKuccl::recv(std::vector<at::Tensor> & /*tensors*/, int /*srcRank*/,
                                            int /*tag*/) {
    TORCH_CHECK(false, "kuccl backend does not support send/recv: the public API is ",
                "scoped to 4 collectives (allgather/allreduce/broadcast/alltoall). ",
                "Use gloo/mpi for point-to-point.");
}

/* ── barrier ──────────────────────────────────────────────
 * 用 1 字节 allreduce 实现屏障（语义等价，全员到达才完成）。
 * KUCCL_BARRIER=store 回退 TCPStore 版（排障用）。 */
c10::intrusive_ptr<Work> BackendKuccl::barrier(const BarrierOptions &) {
    if (!barrier_use_store() && comm_ready_ && getSize() > 1) {
        /* token 必须堆分配（异步期间栈帧会销毁，UCG 还在写它）——
         * 用 tensor 做托管内存，随 Work 保活 */
        auto token = at::empty({1}, at::TensorOptions().dtype(at::kByte));
        token.fill_(1);
        kucclRequest_t req = nullptr;
        if (kucclAllReduce(token.data_ptr(), token.data_ptr(), 1, kucclUint8, kucclSum, comm_,
                           &req) == KUCCL_SUCCESS) {
            return make_async_work(req, comm_, {token}, OpType::BARRIER);
        }
        /* allreduce 启动失败 → 落回 Store 版，保证 barrier 语义可用 */
    }
    kuccl::store_barrier(store_.get(), getRank(), getSize(),
                         "pg_" + std::to_string(barrier_seq_++));
    return c10::make_intrusive<WorkKuccl>(OpType::BARRIER);
}

/* ── 静态工厂方法 ───────────────────────────────────────── */
c10::intrusive_ptr<Backend>
BackendKuccl::createBackendKuccl(const c10::intrusive_ptr<Store> &store, int rank, int size,
                                 const std::chrono::duration<float> &timeout) {
    auto timeout_ms = std::chrono::duration_cast<std::chrono::milliseconds>(timeout);
    return c10::make_intrusive<BackendKuccl>(store, rank, size, timeout_ms);
}

} // namespace c10d

/* ════════════════════════════════════════════════════════════
 * PYBIND11_MODULE — 唯一入口
 * __attribute__((constructor)) 在 backend.hpp 中实现自动注册，
 * 用户只需 `import kuccl` 即可，无需手动 register_backend。
 * ════════════════════════════════════════════════════════════*/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "KUCCL (UCX + UCG) PyTorch Backend V2 — auto-registered on import";
    m.def("create_kuccl_backend", &c10d::BackendKuccl::createBackendKuccl, py::arg("store"),
          py::arg("rank"), py::arg("world_size"), py::arg("timeout"));
}
