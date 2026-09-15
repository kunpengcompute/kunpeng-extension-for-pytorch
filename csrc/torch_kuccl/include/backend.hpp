#ifndef KUCCL_BACKEND_HPP_
#define KUCCL_BACKEND_HPP_

/*
 * KUCCL PyTorch Backend 适配层
 *
 *   import kuccl
 *   dist.init_process_group("kuccl", rank=0, world_size=1)
 *   dist.all_reduce(x)  ← 走 kuccl
 */

#include <torch/python.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include "oob.hpp" // 本包：Store 版 OOB 工具
#include <kuccl.h> // libkuccl：唯一公共头（不透明句柄）

#include <pybind11/chrono.h>

#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace c10d {

/* ── WorkKuccl（异步 Work：持有在飞的 kucclRequest）────────
 *
 * 对齐 c10d 的异步语义：
 *   - 后端算子立即返回 Work（通信在飞），同步/异步的区分在 c10d 层
 *     （async_op=False 时 c10d 调 work.wait()；True 时 Work 给用户）
 *   - wait()/isCompleted() 内部推进请求——循环在 Work 里，
 *     用户永远不用手写循环
 *
 * 异步期间的 tensor 生命周期：Work 持有参与算子的 tensor 引用
 * （tensors_ 成员），防止用户 tensor 在通信完成前被释放。 */
class WorkKuccl : public Work {
public:
    /* 空操作构造（零元素/无请求场景）：立即完成 */
    explicit WorkKuccl(OpType opType = OpType::ALLGATHER)
        : Work(-1, opType), req_(nullptr), done_(true), failed_(false) {
        future_ = c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());
    }

    /* 异步构造：持有在飞请求 + 参与的 tensor（保活）+ comm（flush 用）
     * finisher：完成时需要执行的收尾（如 allgather 列表式的拆回拷贝） */
    WorkKuccl(kucclRequest_t req, kucclComm_t comm, std::vector<at::Tensor> tensors,
              OpType opType = OpType::ALLGATHER, std::function<void()> finisher = nullptr)
        : Work(-1, opType), req_(req), tensors_(std::move(tensors)), finisher_(std::move(finisher)),
          done_(false), failed_(false) {
        (void)comm; /* 不透明化后 Work 不持 worker；comm 仅构造期校验用 */
        future_ = c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());
    }

    ~WorkKuccl() override {
        /* 析构时若用户既没 wait 也没查询完成——兜底同步，
         * 防止请求泄漏（UCG worker 已销毁时 kucclTest 会失败，
         * 此时只能放弃——文档约束：销毁 ProcessGroup 前应等待所有 Work） */
        synchronize();
    }

    bool isCompleted() override {
        poll_once();
        return done_.load(std::memory_order_acquire);
    }

    bool isSuccess() const override { return !failed_.load(std::memory_order_acquire); }

    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override {
        /* 无超时：自旋推进直到完成（g_api_lock 保证多线程查询安全）。
         * 有超时：到点返回当前状态（不视为失败——用户可再 wait）。 */
        if (timeout == kUnsetTimeout) {
            while (!done_.load(std::memory_order_acquire)) {
                poll_once();
            }
        } else {
            auto deadline = std::chrono::steady_clock::now() + timeout;
            while (!done_.load(std::memory_order_acquire)) {
                if (std::chrono::steady_clock::now() >= deadline)
                    break;
                poll_once();
            }
        }
        return isSuccess();
    }

    void synchronize() override {
        while (!done_.load(std::memory_order_acquire)) {
            poll_once();
        }
    }

    void blockCurrentStream() override {}

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
        /* Future 语义：首次查询时若已完成则标记完成 */
        poll_once();
        if (done_.load(std::memory_order_acquire) && !future_marked_) {
            future_->markCompleted(); /* 错误经 isSuccess()/wait() 暴露 */
            future_marked_ = true;
        }
        return future_;
    }

private:
    /* 推进一拍：查询请求状态；完成时收割、执行 finisher、标记 */
    void poll_once() const {
        if (done_.load(std::memory_order_acquire))
            return;
        int flag = 0;
        kucclStatus_t st = KUCCL_STATUS_PENDING;
        kucclRequest_t req = req_;
        kucclResult_t rc = kucclTest(&req, &flag, &st);
        if (rc != KUCCL_SUCCESS && flag == 0) {
            /* 查询失败且未完成：区分"暂时失败"（下次再试）与
             * "致命失败"（comm 已销毁/aborted——继续轮询只会
             * 无限自旋，标记失败退出，对齐 watchdog 语义） */
            if (fail_count_++ > 1000) {
                done_.store(true, std::memory_order_release);
                failed_.store(true, std::memory_order_release);
            }
            return;
        }
        fail_count_ = 0;
        if (flag == 1) {
            req_ = nullptr;
            const bool ok = (st == KUCCL_STATUS_COMPLETE);
            if (ok && finisher_) {
                finisher_(); /* 完成时收尾（如 allgather 拆回拷贝） */
            }
            done_.store(true, std::memory_order_release);
            failed_.store(!ok, std::memory_order_release);
        }
    }

    mutable kucclRequest_t req_;
    std::vector<at::Tensor> tensors_; /* tensor 保活（异步期间不释放） */
    std::function<void()> finisher_;  /* 完成时收尾（列表式拆回等） */
    mutable std::atomic<bool> done_;
    mutable std::atomic<bool> failed_;
    mutable std::atomic<int> fail_count_{0};
    mutable bool future_marked_ = false;
    c10::intrusive_ptr<c10::ivalue::Future> future_;
};

using WorkKucclPtr = c10::intrusive_ptr<WorkKuccl>;

/* ── BackendKuccl（c10d::Backend 实现）─────────────────── */
class BackendKuccl : public Backend {
public:
    BackendKuccl(c10::intrusive_ptr<Store> store, int rank, int world_size,
                 std::chrono::milliseconds timeout);
    ~BackendKuccl() override;

    const std::string getBackendName() const override { return "kuccl"; }

    c10::intrusive_ptr<Work> allgather(std::vector<std::vector<at::Tensor>> &outputTensors,
                                       std::vector<at::Tensor> &inputTensors,
                                       const AllgatherOptions &opts) override;

    c10::intrusive_ptr<Work> _allgather_base(at::Tensor &outputBuffer, at::Tensor &inputBuffer,
                                             const AllgatherOptions &opts) override;

    c10::intrusive_ptr<Work> broadcast(std::vector<at::Tensor> &tensors,
                                       const BroadcastOptions &opts) override;

    c10::intrusive_ptr<Work> allreduce(std::vector<at::Tensor> &tensors,
                                       const AllreduceOptions &opts) override;

    c10::intrusive_ptr<Work> allreduce_coalesced(std::vector<at::Tensor> &tensors,
                                                 const AllreduceCoalescedOptions &opts) override;

    c10::intrusive_ptr<Work> alltoall_base(at::Tensor &outputBuffer, at::Tensor &inputBuffer,
                                           std::vector<int64_t> &outputSplitSizes,
                                           std::vector<int64_t> &inputSplitSizes,
                                           const AllToAllOptions &opts) override;

    c10::intrusive_ptr<Work> send(std::vector<at::Tensor> &tensors, int dstRank, int tag) override;

    c10::intrusive_ptr<Work> recv(std::vector<at::Tensor> &tensors, int srcRank, int tag) override;

    c10::intrusive_ptr<Work> barrier(const BarrierOptions &opts) override;

    /* 安全终止（c10d destroy_process_group 调用链）
     * —— 标记 comm aborted，后续算子立即失败而非卡死/未定义 */
    void shutdown() override;

    static c10::intrusive_ptr<Backend>
    createBackendKuccl(const c10::intrusive_ptr<Store> &store, int rank, int size,
                       const std::chrono::duration<float> &timeout);

    /* auto-register：import kuccl 即自动注册
     *
     * PyTorch 2.9 的签名：register_backend(name, func, extended_api=False, devices=None)
     * ★ devices 必须作为第 4 个参数传入（extended_api 之后）。
     *   若误传到第 3 个位置（extended_api 槽），truthy 列表会使 PyTorch
     *   按扩展 API 约定调用工厂 creator_fn(_DistributedBackendOptions, opts)，
     *   与 createBackendKuccl(store, rank, world_size, timeout) 签名不符
     *   → init_process_group 报 TypeError（PyTorch 2.9 实测坑）。
     *
     * devices=["cpu"] 的作用：
     *   1) 消除 "Device capability unspecified" 告警
     *   2) 避免 cuda 张量被误路由到 kuccl
     *   3) 支持 "cpu:kuccl" 设备前缀写法 */
    static void autoRegister() __attribute__((constructor)) {
        py::object module = py::module::import("torch.distributed");
        py::object register_backend = module.attr("Backend").attr("register_backend");
        py::list devices;
        devices.append("cpu");
        register_backend("kuccl", py::cpp_function(createBackendKuccl), py::bool_(false), devices);
    }

    /* Store → kuccl OOB 回调的桥接上下文
     * public：回调函数（backend.cpp）需要访问其成员 */
    struct StoreOobCtx {
        Store *store = nullptr;
        int rank_hint = 0;
        int ws_hint = 0;
        std::atomic<std::uint64_t> round{0};
    };

private:
    c10::intrusive_ptr<Store> store_;
    kucclComm_t comm_ = nullptr;
    bool comm_ready_ = false;

    StoreOobCtx store_oob_; /* 直接成员（生命周期随 BackendKuccl） */

    std::uint64_t barrier_seq_ = 0;

    /* allgather（列表式）的中转缓冲复用：免去每次调用 at::empty 的
     * 大块 malloc/munmap + 首次触碰缺页。
     * 复用条件：dtype 一致且容量足够；跨 dtype/扩容时自动重建。 */
    at::Tensor allgather_scratch_;

    /* torch dtype → kucclDataType_t 映射 */
    static kucclDataType_t torchToKucclDtype(c10::ScalarType st);
    /* torch ReduceOp → kucclRedOp_t 映射 */
    static kucclRedOp_t torchToKucclRedOp(const ReduceOp &op);
};

} /* namespace c10d */

#endif /* KUCCL_BACKEND_HPP_ */
