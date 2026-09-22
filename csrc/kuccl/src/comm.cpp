/*
 * KuCCL 通信域管理实现（骨架层，后端分发）
 *
 * 实现 4 个通信域接口：
 *   kucclGetUniqueId       — 生成全局唯一 ID
 *   kucclCommInitRankConfig — 创建通信域（后端解析 + process_init + group_init）
 *   kucclCommDestroy       — 销毁通信域（comm_purge + group_fini）
 *   kucclCommSplit         — 通信域切分
 * 以及 kucclCommAbort / kucclFlushComm 与进程级 finalize。
 *
 * 后端选择：config.backend > 环境变量 KUCCL_BACKEND_TYPE
 * > 默认 "xucg"。底层实现由 xucg 后端承接。
 *
 * ★ extern "C" 必须同时包裹声明与实现：只包声明时，实现按 C++ 修饰名
 *   编译，链接时 undefined symbol。
 */

#include "comm_impl.hpp"
#include "backend/backend.hpp"
#include "log.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <new>
#include <unistd.h>
#include <vector>

namespace kuccl {
namespace detail {

namespace {
/* 当前活跃后端（首个成功建组的后端；finalize 用） */
const backend::Ops *g_active_backend = nullptr;
} /* anonymous namespace */

void set_active_backend(const backend::Ops *be) {
    g_active_backend = be;
}
const backend::Ops *active_backend() {
    return g_active_backend;
}

/* OOB allgather 快捷封装：走 config.oob_allgather 回调 */
int oob_exchange(const kucclConfig_t *cfg, const void *sendbuf, void *recvbuf, int count) {
    if (!cfg || !cfg->oob_allgather)
        return -1;
    return cfg->oob_allgather(sendbuf, recvbuf, count, cfg->oob_ctx);
}

} // namespace detail
} // namespace kuccl

namespace {

std::atomic<uint64_t> g_unique_id_counter{1};

/* SplitMix64 — uniqueId 生成 */
uint64_t kuccl_mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

} /* anonymous namespace */

/* ════════════════════════════════════════════════════════════
 * 对外接口实现
 * ════════════════════════════════════════════════════════════*/
extern "C" {

KuCCL_API kucclResult_t kucclGetUniqueId(kucclUniqueId *uniqueId) {
    if (!uniqueId)
        return KuCCL_FAIL;

    /* 用 timestamp + pid + counter 生成 128 字节唯一 ID */
    const uint64_t now =
        static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    const uint64_t pid = static_cast<uint64_t>(static_cast<uint32_t>(::getpid()));
    const uint64_t cnt = g_unique_id_counter.fetch_add(1, std::memory_order_relaxed);

    uint64_t *p = reinterpret_cast<uint64_t *>(uniqueId->data);
    std::memset(uniqueId->data, 0, sizeof(uniqueId->data));
    p[0] = 0x4B5543434C323030ULL; /* "KuCCL200" magic */
    p[1] = now;
    p[2] = pid;
    p[3] = cnt;
    p[4] = kuccl_mix64(now ^ (pid << 32));
    p[5] = kuccl_mix64(cnt * 0x9e3779b97f4a7c15ULL);
    p[6] = kuccl_mix64(now + pid + cnt);
    p[7] = kuccl_mix64(p[4] ^ p[5] ^ p[6]);
    for (int i = 8; i < 16; ++i) {
        p[i] = kuccl_mix64(static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL + p[7]);
    }
    return KuCCL_SUCCESS;
}

/* ════════════════════════════════════════════════════════════
 * 后端解析：config.backend > env KUCCL_BACKEND_TYPE > 默认 xucg
 * ════════════════════════════════════════════════════════════*/
static const kuccl::backend::Ops *kuccl_resolve_backend(const kucclConfig_t *config, int rank) {
    const char *name = nullptr;
    if (config != nullptr && config->backend != nullptr && config->backend[0] != '\0') {
        name = config->backend;
    } else {
        name = std::getenv("KUCCL_BACKEND_TYPE");
        if (name == nullptr || name[0] == '\0')
            name = nullptr;
    }

    const kuccl::backend::Ops *be = kuccl::backend::get(name);
    if (be == nullptr) {
        kuccl::detail::err_record(
            "kucclCommInitRankConfig: unknown backend '%s' (support: xucg/kupl)",
            name != nullptr ? name : "(null)");
        KuCCL_LOG_ERROR(rank, "KUCCL_BACKEND_TYPE='%s' 无法识别（支持: xucg；kupl 预留）",
                        name != nullptr ? name : "(null)");
        return nullptr;
    }
    return be;
}

KuCCL_API kucclResult_t kucclCommInitRankConfig(kucclComm_t *comm, int nranks, kucclUniqueId commId,
                                                int rank, kucclConfig_t *config) {
    if (!comm || nranks <= 0 || rank < 0 || rank >= nranks)
        return KuCCL_FAIL;
    if (!config || !config->oob_allgather) {
        KuCCL_LOG_ERROR(rank, "kucclCommInitRankConfig: config->oob_allgather is required");
        return KuCCL_FAIL;
    }

    /* 建组触碰后端进程级单例：与算子入口同一把锁串行化
     *（递归锁：CommSplit 内部重入本函数） */
    std::lock_guard<std::recursive_mutex> api_lock(kuccl::detail::g_api_lock);

    /* 后端解析 */
    const kuccl::backend::Ops *be = kuccl_resolve_backend(config, rank);
    if (be == nullptr)
        return KuCCL_FAIL;
    (void)commId; /* commId 参与全局组区分（当前用 bootstrap token 保证唯一性）*/

    auto *c = new (std::nothrow) kucclComm();
    if (!c)
        return KuCCL_FAIL;
    std::memset(c, 0, sizeof(kucclComm));
    c->rank = rank;
    c->nranks = nranks;
    c->oob_config = *config;
    c->has_oob = 1;
    c->be = be;

    /* 后端进程级一次性初始化（首次调用才真正执行）*/
    if (be->process_init(c) != KuCCL_SUCCESS) {
        be->group_fini(c); /* 回收 process_init 已分配的私有状态（幂等） */
        delete c;
        return KuCCL_FAIL;
    }

    /* 后端建组（bootstrap 交换 + rank 映射 + 底层组创建）*/
    if (be->group_init(c) != KuCCL_SUCCESS) {
        be->group_fini(c); /* 回收部分初始化的私有状态 */
        delete c;
        return KuCCL_FAIL;
    }

    kuccl::detail::set_active_backend(be);
    *comm = c;
    return KuCCL_SUCCESS;
}

KuCCL_API kucclResult_t kucclCommDestroy(kucclComm_t comm) {
    if (!comm)
        return KuCCL_SUCCESS;
    std::lock_guard<std::recursive_mutex> api_lock(kuccl::detail::g_api_lock);
    /* 先清后端私有状态（持久化请求池持有组句柄引用，必须先于组销毁）*/
    comm->be->comm_purge(comm);
    comm->be->group_fini(comm);
    delete comm;
    return KuCCL_SUCCESS;
}

/* ════════════════════════════════════════════════════════════
 * kucclCommAbort —— 安全终止
 *
 * 语义：
 *   - 立即标记 aborted：后续所有算子返回 FAIL（带错误信息）
 *   - 不保证在飞请求数据完整（故障恢复用，非正常销毁）
 *   - 幂等、线程安全（atomic + lock）
 *   - 正常退出仍用 kucclCommDestroy（Abort 后也可再 Destroy 清资源）
 * ════════════════════════════════════════════════════════════*/
KuCCL_API kucclResult_t kucclCommAbort(kucclComm_t comm) {
    if (!comm)
        return KuCCL_SUCCESS;
    std::lock_guard<std::recursive_mutex> api_lock(kuccl::detail::g_api_lock);

    /* 幂等：已 aborted 直接成功返回 */
    if (comm->aborted.load(std::memory_order_acquire))
        return KuCCL_SUCCESS;

    comm->aborted.store(1, std::memory_order_release);
    kuccl::detail::err_record("comm aborted (rank=%d, nranks=%d): subsequent calls will fail",
                              comm->rank, comm->nranks);
    KuCCL_LOG_WARN(comm->rank, "kucclCommAbort: comm marked ABORTED");

    /* 在飞请求：尽力清理后端私有状态（非持久化的由持有方 Test 时
     * 发现 aborted 状态报错——不强行 cancel 在飞请求，
     * 尽力而为语义） */
    comm->be->comm_purge(comm);
    return KuCCL_SUCCESS;
}

KuCCL_API kucclResult_t kucclFlushComm(kucclComm_t comm) {
    if (!comm)
        return KuCCL_FAIL;
    if (comm->aborted.load(std::memory_order_acquire)) {
        kuccl::detail::err_record("kucclFlushComm: comm is ABORTED");
        return KuCCL_FAIL;
    }
    return comm->be->flush(comm);
}

KuCCL_API kucclResult_t kucclCommSplit(kucclComm_t comm, int color, int key, kucclComm_t *newcomm,
                                       kucclConfig_t *config) {
    if (!comm || !newcomm || !comm->has_oob)
        return KuCCL_FAIL;

    std::lock_guard<std::recursive_mutex> api_lock(kuccl::detail::g_api_lock);

    /* 父 comm 已 abort 则拒绝切分（Split 走父 comm 的 OOB，
     * aborted 后行为未定义） */
    if (comm->aborted.load(std::memory_order_acquire)) {
        kuccl::detail::err_record("kucclCommSplit: parent comm is ABORTED");
        return KuCCL_FAIL;
    }

    if (color < 0) {
        KuCCL_LOG_INFO(comm->rank, "kucclCommSplit: color<0 (不加入任何子组)");
        return KuCCL_FAIL;
    }
    (void)key; /* key 影响子组内排序；成员序按父 comm 的 rank 序（OOB 顺序） */

    /* OOB 交换所有进程的 color（走父 comm 的 OOB） */
    const int ws = comm->nranks;
    std::vector<int32_t> colors(static_cast<std::size_t>(ws), -1);
    const int32_t my_color = static_cast<int32_t>(color);
    if (kuccl::detail::oob_exchange(&comm->oob_config, &my_color, colors.data(),
                                    static_cast<int>(sizeof(int32_t))) != 0) {
        KuCCL_LOG_ERROR(comm->rank, "kucclCommSplit: OOB color exchange FAILED");
        return KuCCL_FAIL;
    }

    /* 收集同 color 的进程（OOB 顺序 = 父 comm rank 序，所有成员一致） */
    std::vector<int32_t> members;
    for (int i = 0; i < ws; ++i) {
        if (colors[static_cast<std::size_t>(i)] == my_color) {
            members.push_back(static_cast<int32_t>(i));
        }
    }
    if (members.empty()) {
        KuCCL_LOG_ERROR(comm->rank, "kucclCommSplit: empty group");
        return KuCCL_FAIL;
    }

    /* 在新组内的 rank */
    int new_rank = -1;
    for (std::size_t i = 0; i < members.size(); ++i) {
        if (members[i] == comm->rank) {
            new_rank = static_cast<int>(i);
            break;
        }
    }
    if (new_rank < 0) {
        KuCCL_LOG_ERROR(comm->rank, "kucclCommSplit: self not in members");
        return KuCCL_FAIL;
    }

    /* 创建新 comm（复用 InitRankConfig 建组逻辑；后端继承父 comm） */
    kucclUniqueId dummy_id;
    kucclGetUniqueId(&dummy_id);
    kucclConfig_t cfg = (config != nullptr) ? *config : comm->oob_config;
    return kucclCommInitRankConfig(newcomm, static_cast<int>(members.size()), dummy_id, new_rank,
                                   &cfg);
}

} /* extern "C" — 通信域接口结束 */

/* ════════════════════════════════════════════════════════════
 * 进程级终结
 * ════════════════════════════════════════════════════════════*/
namespace kuccl {

void finalize() {
    const backend::Ops *be = detail::active_backend();
    if (be != nullptr) {
        be->process_finalize();
    }
}

} /* namespace kuccl */
