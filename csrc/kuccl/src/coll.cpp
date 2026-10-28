/*
 * KuCCL Coll 实现 —— 集合算子骨架（后端分发层）
 *
 * 本文件是后端无关的骨架：参数校验、串行化锁、abort 快速通道、
 * 异步请求外壳与看门狗，集合动作全部委托给通信域绑定的后端
 * （kucclComm::be，见 backend/backend.hpp）。
 *
 * 公开集合接口（4 个）：
 *   kucclAllGather / kucclAllReduce / kucclBroadcast / kucclAllToAll
 *
 * 异步模型：算子内部走「(缓存)dt/op → request_init → start」后立即返回，
 * 用户调用 kucclTest 查询完成，完成时统一回收。
 */

#include "kuccl.h"

#include "comm_impl.hpp"
#include "backend/backend.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>

namespace {

/* 看门狗：KUCCL_TEST_TIMEOUT_MS=N（默认 0 = 无限等待）。
 * 超时返回 KuCCL_FAIL + 错误信息，防止对端掉线卡死整进程。 */
inline int64_t test_timeout_ms()
{
    static const int64_t v = [] {
        const char *e = std::getenv("KUCCL_TEST_TIMEOUT_MS");
        return e ? static_cast<int64_t>(std::atoll(e)) : 0;
    }();
    return v;
}

/* 算子名（诊断信息用；与公开接口一一对应） */
const char *coll_op_name(kuccl::backend::CollOp op)
{
    using kuccl::backend::CollOp;
    switch (op) {
        case CollOp::AllGather:
            return "allgather";
        case CollOp::AllReduce:
            return "allreduce";
        case CollOp::Broadcast:
            return "broadcast";
        case CollOp::AllToAll:
            return "alltoall";
    }
    return "op";
}

/* abort 快速通道：aborted 后拒绝一切新操作。
 * 调用方必须已持 g_api_lock（锁在公开函数体内获取，覆盖整个函数体）。 */
kucclResult_t coll_entry_guard(kucclComm_t comm, const char *who)
{
    if (!comm) {
        kuccl::detail::err_record("%s: comm is NULL", who);
        return KuCCL_FAIL;
    }
    if (comm->aborted.load(std::memory_order_acquire)) {
        kuccl::detail::err_record("%s: comm is ABORTED", who);
        return KuCCL_FAIL;
    }
    return KuCCL_SUCCESS;
}

} /* namespace */

/* ============================================================================
 * C 接口实现（extern "C"，与 kuccl.h 声明保持 C 链接）
 * ==========================================================================*/
extern "C" {

/* ----------------------------------------------------------------------------
 * kucclAllGather —— 每进程贡献 sendcount 个元素，所有进程收到全部拼接
 * --------------------------------------------------------------------------*/
KuCCL_API kucclResult_t kucclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount, kucclDataType_t datatype,
                                       kucclComm_t comm, kucclRequest_t *request)
{
    if (!sendbuff || !recvbuff || sendcount == 0) {
        kuccl::detail::err_record("kucclAllGather: sendbuff/recvbuf 为空或 sendcount=0 (sendcount=%zu)", sendcount);
        return KuCCL_FAIL;
    }
    std::lock_guard<std::recursive_mutex> api_lock(kuccl::detail::g_api_lock);
    kuccl::detail::err_clear(); /* 本轮错误从零开始计 */
    if (coll_entry_guard(comm, __func__) != KuCCL_SUCCESS)
        return KuCCL_FAIL;

    kuccl::backend::CollParams p;
    p.sendbuff = sendbuff;
    p.recvbuff = recvbuff;
    p.count = sendcount;
    p.datatype = datatype;

    kucclRequest *req = new kucclRequest();
    req->be = comm->be;
    req->op_name = coll_op_name(kuccl::backend::CollOp::AllGather);
    req->t0 = std::chrono::steady_clock::now();
    kucclResult_t r = comm->be->coll_start(comm, kuccl::backend::CollOp::AllGather, &p, req);
    if (r != KuCCL_SUCCESS) {
        delete req;
        return r; /* 透传 backend 的映射错误码 */
    }
    *request = req;
    return KuCCL_SUCCESS;
}

/* ----------------------------------------------------------------------------
 * kucclAllReduce —— 所有进程数据按 reduction_op 归约，结果写入 recvbuff
 *   ★必须用真实 dtype，否则逐字节归约出错
 * --------------------------------------------------------------------------*/
KuCCL_API kucclResult_t kucclAllReduce(void *sendbuff, void *recvbuff, size_t count, kucclDataType_t datatype,
                                       kucclRedOp_t reduction_op, kucclComm_t comm, kucclRequest_t *request)
{
    if (!sendbuff || !recvbuff || count == 0) {
        kuccl::detail::err_record("%s: sendbuff/recvbuf 为空或 count=0 (count=%zu)", __func__, count);
        return KuCCL_FAIL;
    }
    std::lock_guard<std::recursive_mutex> api_lock(kuccl::detail::g_api_lock);
    kuccl::detail::err_clear(); /* 本轮错误从零开始计 */
    if (coll_entry_guard(comm, __func__) != KuCCL_SUCCESS)
        return KuCCL_FAIL;

    kuccl::backend::CollParams p;
    p.sendbuff = sendbuff;
    p.recvbuff = recvbuff;
    p.count = count;
    p.datatype = datatype;
    p.redop = reduction_op;

    kucclRequest *req = new kucclRequest();
    req->be = comm->be;
    req->op_name = coll_op_name(kuccl::backend::CollOp::AllReduce);
    req->t0 = std::chrono::steady_clock::now();
    kucclResult_t r = comm->be->coll_start(comm, kuccl::backend::CollOp::AllReduce, &p, req);
    if (r != KuCCL_SUCCESS) {
        delete req;
        return r; /* 透传 backend 的映射错误码 */
    }
    *request = req;
    return KuCCL_SUCCESS;
}

/* ----------------------------------------------------------------------------
 * kucclBroadcast —— root 进程数据广播给所有进程
 * --------------------------------------------------------------------------*/
KuCCL_API kucclResult_t kucclBroadcast(const void *sendbuff, void *recvbuff, size_t count, kucclDataType_t datatype,
                                       int root, kucclComm_t comm, kucclRequest_t *request)
{
    if (!sendbuff || !recvbuff || count == 0) {
        kuccl::detail::err_record("%s: sendbuff/recvbuf 为空或 count=0 (count=%zu)", __func__, count);
        return KuCCL_FAIL;
    }
    if (root < 0 || root >= comm->nranks) {
        kuccl::detail::err_record("kucclBroadcast: root 越界 (root=%d nranks=%d)", root, comm->nranks);
        return KuCCL_FAIL;
    }
    std::lock_guard<std::recursive_mutex> api_lock(kuccl::detail::g_api_lock);
    kuccl::detail::err_clear(); /* 本轮错误从零开始计 */
    if (coll_entry_guard(comm, __func__) != KuCCL_SUCCESS)
        return KuCCL_FAIL;

    kuccl::backend::CollParams p;
    p.sendbuff = sendbuff;
    p.recvbuff = recvbuff;
    p.count = count;
    p.datatype = datatype;
    p.root = root;

    kucclRequest *req = new kucclRequest();
    req->be = comm->be;
    req->op_name = coll_op_name(kuccl::backend::CollOp::Broadcast);
    req->t0 = std::chrono::steady_clock::now();
    kucclResult_t r = comm->be->coll_start(comm, kuccl::backend::CollOp::Broadcast, &p, req);
    if (r != KuCCL_SUCCESS) {
        delete req;
        return r; /* 透传 backend 的映射错误码（如 NOT_FOUND=-14） */
    }
    *request = req;
    return KuCCL_SUCCESS;
}

/* ----------------------------------------------------------------------------
 * kucclAllToAll —— 所有进程之间等长全交换
 * --------------------------------------------------------------------------*/
KuCCL_API kucclResult_t kucclAllToAll(const void *sendbuff, void *recvbuff, size_t count, kucclDataType_t datatype,
                                      kucclComm_t comm, kucclRequest_t *request)
{
    if (!sendbuff || !recvbuff || count == 0) {
        kuccl::detail::err_record("%s: sendbuff/recvbuf 为空或 count=0 (count=%zu)", __func__, count);
        return KuCCL_FAIL;
    }
    std::lock_guard<std::recursive_mutex> api_lock(kuccl::detail::g_api_lock);
    kuccl::detail::err_clear(); /* 本轮错误从零开始计 */
    if (coll_entry_guard(comm, __func__) != KuCCL_SUCCESS)
        return KuCCL_FAIL;

    kuccl::backend::CollParams p;
    p.sendbuff = sendbuff;
    p.recvbuff = recvbuff;
    p.count = count;
    p.datatype = datatype;

    kucclRequest *req = new kucclRequest();
    req->be = comm->be;
    req->op_name = coll_op_name(kuccl::backend::CollOp::AllToAll);
    req->t0 = std::chrono::steady_clock::now();
    kucclResult_t r = comm->be->coll_start(comm, kuccl::backend::CollOp::AllToAll, &p, req);
    if (r != KuCCL_SUCCESS) {
        delete req;
        return r; /* 透传 backend 的映射错误码 */
    }
    *request = req;
    return KuCCL_SUCCESS;
}

/* kucclTest —— 查询请求完成：后端先 test（就绪即返回）再推进；
 * 完成/出错时后端回收私有状态，这里释放请求外壳。 */
KuCCL_API kucclResult_t kucclTest(kucclRequest_t *request, int *flag, kucclStatus_t *status)
{
    if (!request || !*request || !flag || !status)
        return KuCCL_FAIL;
    /* test/progress 都触碰共享后端资源：与算子入口同一把锁串行化 */
    std::lock_guard<std::recursive_mutex> api_lock(kuccl::detail::g_api_lock);
    kuccl::detail::err_clear(); /* 本轮错误从零开始计 */

    kucclRequest *req = *request;

    /* 看门狗：超时返回失败（含诊断信息）——仅当配置了超时 */
    const int64_t timeout = test_timeout_ms();
    if (timeout > 0) {
        auto waited =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - req->t0).count();
        if (waited > timeout) {
            kuccl::detail::err_record("kucclTest: watchdog timeout after %lld ms (op=%s) — "
                                      "peer likely dead or comm aborted",
                                      static_cast<long long>(waited), req->op_name);
            req->be->req_cancel(req);
            *flag = 1;
            *status = KuCCL_STATUS_ERROR;
            delete req;
            *request = nullptr;
            return KuCCL_FAIL;
        }
    }

    /* 后端推进（test + progress + 完成回收） */
    const kuccl::backend::Ops *be = req->be;
    if (be->req_test(req, flag, status) != KuCCL_SUCCESS) {
        *flag = 1;
        *status = KuCCL_STATUS_ERROR;
    }

    if (*flag) {
        delete req;
        *request = nullptr;
    }
    return KuCCL_SUCCESS;
}

} /* extern "C" */
