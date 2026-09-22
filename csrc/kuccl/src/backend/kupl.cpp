/*
 * kupl 后端 —— 单机共享内存快路径（KUPL shm）
 *
 * shm 不跨节点：kupl 后端仅用于单机，跨节点走 xucg。
 * 选择方式：KUCCL_BACKEND_TYPE=kupl 或 kucclConfig_t.backend="kupl"。
 *
 * 底层走 KUPL 公共 API（kupl.h，kupl_export 符号）：
 *   kupl_shm_comm_create / kupl_shm_comm_destroy   通信域（OOB 回调注入）
 *   kupl_shm_win_alloc / kupl_shm_win_free          共享窗口（合作式分配）
 *   kupl_shm_allreduce_init / kupl_shm_bcast_init /
 *   kupl_shm_alltoall_init + request_start/free     集合算子
 *
 * 封装风格：
 *   - OOB 回调由调用方注入（kucclConfig_t 适配为 kupl_shm_oob_cb_t）
 *   - 算子入口做"参数准备 + 校验 + 分发"，动作在 KUPL 内部完成
 *
 * 数据面：
 *   KUPL 要求集合 buffer 位于 kupl_shm_win 内（find_win 校验），而用户
 *   buffer 是普通内存——因此本后端为每个通信域维护一个惰性增长的
 *   staging 窗口，布局 [send 区 | recv 区]，数据 memcpy 进出：
 *     AllReduce : send→窗口，start（同步），完成后拷回 recv
 *     Broadcast : root 拷入窗口（init 内部 fence 保证时序），完成后拷回
 *     AllToAll  : send→窗口，start，完成后拷回
 *     AllGather : 转换为等长 alltoall（每槽填自己的块），完成后拷回
 *   窗口按需翻倍增长；增长决策只依赖集合参数（全员一致），无需额外 OOB。
 *
 * 语义差异：kupl 的 request_start 是同步驱动（算子内部 fence 自旋），
 * request_wait 为空操作——因此本后端的"异步"表现为：coll_start 内完成
 * 数据面动作，首次 kucclTest 即 COMPLETE。算子死锁会阻塞在 coll_start
 * 内部（API 锁内），Test 层看门狗对此无效，属 KUPL 同步模型的固有限制。
 */

#include "backend.hpp" /* err_record / g_api_lock（经 comm_impl.hpp）*/

#include <cstring>
#include <new>
#include <vector>

#if defined(KuCCL_HAVE_KUPL)

#include <kupl.h>

#include <unistd.h>

namespace kuccl {
namespace backend {

namespace {

/* ── 后端私有状态 ─────────────────────────────────────────── */

/* 通信域私有状态（挂在 kucclComm::be_ctx） */
struct KuplComm {
    kucclComm *owner = nullptr; /* 反指宿主（OOB 回调 cookie） */
    kupl_shm_comm_h comm = nullptr;
    kupl_shm_win_h scratch = nullptr; /* staging 窗口：[send 区 | recv 区] */
    void *scratch_base = nullptr;
    size_t scratch_half = 0; /* 单区字节数（页对齐） */
};

/* 请求私有状态（挂在 kucclRequest::priv） */
struct KuplReq {
    kupl_shm_request_h req = nullptr;
    void *user_dst = nullptr;      /* 完成后拷回目的地（nullptr=无拷回） */
    const void *win_src = nullptr; /* 拷回源（窗口 recv 区） */
    size_t copy_bytes = 0;
};

/* OOB 适配：kuccl 回调（config.oob_allgather）→ kupl 签名 */
int kupl_oob_allgather_cb(const void *sendbuf, void *recvbuf, int size, void *group,
                          kupl_shm_datatype_t /*datatype*/) {
    auto *kc = static_cast<KuplComm *>(group);
    if (kc == nullptr || kc->owner == nullptr)
        return -1;
    return detail::oob_exchange(&kc->owner->oob_config, sendbuf, recvbuf, size);
}

/* OOB barrier：用 1 字节 OOB allgather 模拟（语义：全员到齐） */
int kupl_oob_barrier_cb(void *group) {
    auto *kc = static_cast<KuplComm *>(group);
    if (kc == nullptr || kc->owner == nullptr)
        return -1;
    const char send_byte = 0;
    std::vector<char> recv(static_cast<size_t>(kc->owner->nranks), 1);
    return detail::oob_exchange(&kc->owner->oob_config, &send_byte, recv.data(), 1);
}

/* kucclDataType_t → kupl_shm_datatype_t
 * （Int32/Uint32 同映射 INT、Int64/Uint64 同映射 LONG：
 *   SUM/MIN/MAX 的按位语义在补码下一致；有符号性差异见 README 说明） */
kupl_shm_datatype_t map_datatype(kucclDataType_t dt) {
    switch (dt) {
    case kucclInt8:
    case kucclUint8:
        return KUPL_SHM_DATATYPE_CHAR;
    case kucclInt32:
    case kucclUint32:
        return KUPL_SHM_DATATYPE_INT;
    case kucclInt64:
    case kucclUint64:
        return KUPL_SHM_DATATYPE_LONG;
    case kucclFloat:
        return KUPL_SHM_DATATYPE_FLOAT;
    case kucclDouble:
        return KUPL_SHM_DATATYPE_DOUBLE;
    }
    return KUPL_SHM_DATATYPE_CHAR;
}

kupl_shm_reduce_op_t map_redop(kucclRedOp_t op) {
    switch (op) {
    case kucclSum:
        return KUPL_RD_ADD;
    case kucclMin:
        return KUPL_RD_MIN;
    case kucclMax:
        return KUPL_RD_MAX;
    }
    return KUPL_RD_ADD;
}

size_t kupl_dt_size(kucclDataType_t dt) {
    switch (dt) {
    case kucclInt8:
    case kucclUint8:
        return 1;
    case kucclInt32:
    case kucclUint32:
    case kucclFloat:
        return 4;
    case kucclInt64:
    case kucclUint64:
    case kucclDouble:
        return 8;
    }
    return 0;
}

size_t align_page(size_t n) {
    const size_t page = 4096;
    return (n + page - 1) / page * page;
}

/* staging 窗口惰性增长（容量不足则重建为 2×need）。
 * 增长决策只依赖集合参数（count 全员一致），所有进程同步走到同一分支，
 * win_alloc/win_free 的合作式语义（全员同时调用）天然满足。 */
kucclResult_t ensure_scratch(KuplComm *kc, size_t need_bytes) {
    const size_t half = align_page(need_bytes);
    if (kc->scratch != nullptr && kc->scratch_half >= half)
        return KuCCL_SUCCESS;
    if (kc->scratch != nullptr) {
        kupl_shm_win_free(kc->scratch);
        kc->scratch = nullptr;
        kc->scratch_base = nullptr;
        kc->scratch_half = 0;
    }
    void *base = nullptr;
    kupl_shm_win_h win = nullptr;
    if (kupl_shm_win_alloc(half * 2, kc->comm, &base, &win) != 0 || base == nullptr ||
        win == nullptr) {
        detail::err_record("kupl: staging win_alloc(%zu) FAILED", half * 2);
        return KuCCL_FAIL;
    }
    kc->scratch = win;
    kc->scratch_base = base;
    kc->scratch_half = half;
    return KuCCL_SUCCESS;
}

inline void *scratch_send(KuplComm *kc) {
    return kc->scratch_base;
}
inline void *scratch_recv(KuplComm *kc) {
    return static_cast<char *>(kc->scratch_base) + kc->scratch_half;
}

} /* anonymous namespace */

/* ════════════════════════════════════════════════════════════
 * 生命周期钩子
 * ════════════════════════════════════════════════════════════*/

static kucclResult_t kupl_process_init(kucclComm * /*comm*/) {
    /* 无进程级前置：kupl_shm_init 由 comm_create / 各 init 惰性触发 */
    return KuCCL_SUCCESS;
}

static void kupl_process_finalize() {}

static kucclResult_t kupl_group_init(kucclComm *comm) {
    const int rank = comm->rank;

    auto *kc = new (std::nothrow) KuplComm();
    if (kc == nullptr) {
        detail::err_record("kupl: alloc group ctx FAILED");
        return KuCCL_FAIL;
    }
    kc->owner = comm;
    comm->be_ctx = kc;

    kupl_shm_oob_cb_t cbs;
    cbs.oob_allgather = kupl_oob_allgather_cb;
    cbs.oob_barrier = kupl_oob_barrier_cb;

    /* comm_create 内部惰性做 kupl_shm_init，并通过注入的 OOB 回调
     * 收集组员信息（group cookie = kc） */
    if (kupl_shm_comm_create(comm->nranks, rank, static_cast<int>(::getpid()), &cbs, kc,
                             &kc->comm) != 0 ||
        kc->comm == nullptr) {
        detail::err_record("kupl: kupl_shm_comm_create FAILED");
        KuCCL_LOG_ERROR(rank, "kupl_shm_comm_create FAILED");
        delete kc;
        comm->be_ctx = nullptr;
        return KuCCL_FAIL;
    }

    KuCCL_LOG_INFO(rank, "kupl shm comm created (nranks=%d)", comm->nranks);
    return KuCCL_SUCCESS;
}

static void kupl_group_fini(kucclComm *comm) {
    auto *kc = static_cast<KuplComm *>(comm->be_ctx);
    if (kc == nullptr)
        return;
    if (kc->scratch != nullptr) {
        kupl_shm_win_free(kc->scratch);
        kc->scratch = nullptr;
    }
    if (kc->comm != nullptr) {
        kupl_shm_comm_destroy(kc->comm);
        kc->comm = nullptr;
    }
    delete kc;
    comm->be_ctx = nullptr;
}

/* ════════════════════════════════════════════════════════════
 * 集合算子：参数准备 → staging → kupl request（start 同步驱动）
 * ════════════════════════════════════════════════════════════*/

static kucclResult_t kupl_coll_start(kucclComm *comm, CollOp op, const CollParams *p,
                                     kucclRequest *req) {
    const int rank = comm->rank;
    auto *kc = static_cast<KuplComm *>(comm->be_ctx);

    auto *priv = new (std::nothrow) KuplReq();
    if (priv == nullptr) {
        detail::err_record("kupl: alloc request state FAILED");
        return KuCCL_FAIL;
    }

    kupl_shm_request_h kreq = nullptr;
    const size_t dt_size = kupl_dt_size(p->datatype);

    switch (op) {
    case CollOp::AllReduce: {
        const size_t bytes = p->count * dt_size;
        if (bytes == 0 || bytes > static_cast<size_t>(INT32_MAX) ||
            ensure_scratch(kc, bytes) != KuCCL_SUCCESS) {
            detail::err_record("kupl allreduce: staging FAILED");
            delete priv;
            return KuCCL_FAIL;
        }
        void *send_r = scratch_send(kc);
        void *recv_r = scratch_recv(kc);
        std::memcpy(send_r, p->sendbuff, bytes);
        if (kupl_shm_allreduce_init(send_r, recv_r, static_cast<int>(p->count),
                                    map_datatype(p->datatype), map_redop(p->redop), kc->comm,
                                    &kreq) != 0) {
            detail::err_record("kupl: allreduce_init FAILED");
            delete priv;
            return KuCCL_FAIL;
        }
        if (kupl_shm_request_start(kreq) != 0) {
            detail::err_record("kupl: allreduce start FAILED");
            kupl_shm_request_free(kreq);
            delete priv;
            return KuCCL_FAIL;
        }
        priv->user_dst = p->recvbuff;
        priv->win_src = recv_r;
        priv->copy_bytes = bytes;
        break;
    }
    case CollOp::Broadcast: {
        const size_t bytes = p->count * dt_size;
        if (bytes == 0 || bytes > static_cast<size_t>(INT32_MAX) ||
            ensure_scratch(kc, bytes) != KuCCL_SUCCESS) {
            detail::err_record("kupl bcast: staging FAILED");
            delete priv;
            return KuCCL_FAIL;
        }
        void *buf_r = scratch_recv(kc);
        /* root 先拷入；bcast_init 内部对 buffer 所在窗口做 fence，
         * 保证 root 拷入 happens-before 各 rank 的算法读取 */
        if (comm->rank == p->root) {
            std::memcpy(buf_r, p->sendbuff, bytes);
        }
        if (kupl_shm_bcast_init(buf_r, static_cast<int>(p->count), map_datatype(p->datatype),
                                p->root, kc->comm, &kreq) != 0) {
            detail::err_record("kupl: bcast_init FAILED");
            delete priv;
            return KuCCL_FAIL;
        }
        if (kupl_shm_request_start(kreq) != 0) {
            detail::err_record("kupl: bcast start FAILED");
            kupl_shm_request_free(kreq);
            delete priv;
            return KuCCL_FAIL;
        }
        priv->user_dst = p->recvbuff;
        priv->win_src = buf_r;
        priv->copy_bytes = bytes;
        break;
    }
    case CollOp::AllToAll: {
        const size_t per_rank = p->count * dt_size;
        const size_t total = per_rank * static_cast<size_t>(comm->nranks);
        if (per_rank == 0 || total > static_cast<size_t>(INT32_MAX) ||
            ensure_scratch(kc, total) != KuCCL_SUCCESS) {
            detail::err_record("kupl alltoall: staging FAILED");
            delete priv;
            return KuCCL_FAIL;
        }
        void *send_r = scratch_send(kc);
        void *recv_r = scratch_recv(kc);
        std::memcpy(send_r, p->sendbuff, total);
        if (kupl_shm_alltoall_init(send_r, static_cast<int>(p->count), map_datatype(p->datatype),
                                   recv_r, static_cast<int>(p->count), map_datatype(p->datatype),
                                   kc->comm, &kreq) != 0) {
            detail::err_record("kupl: alltoall_init FAILED");
            delete priv;
            return KuCCL_FAIL;
        }
        if (kupl_shm_request_start(kreq) != 0) {
            detail::err_record("kupl: alltoall start FAILED");
            kupl_shm_request_free(kreq);
            delete priv;
            return KuCCL_FAIL;
        }
        priv->user_dst = p->recvbuff;
        priv->win_src = recv_r;
        priv->copy_bytes = total;
        break;
    }
    case CollOp::AllGather: {
        /* 等长 alltoall 转换：每槽填自己的块（CHAR 按字节计数）。
         * 发送总量 = 接收总量 = nranks × chunk，与 allgather 一次
         * 线性搬运的流量同级 */
        const size_t chunk = p->count * dt_size;
        const size_t total = chunk * static_cast<size_t>(comm->nranks);
        if (chunk == 0 || total > static_cast<size_t>(INT32_MAX) ||
            ensure_scratch(kc, total) != KuCCL_SUCCESS) {
            detail::err_record("kupl allgather: staging FAILED");
            delete priv;
            return KuCCL_FAIL;
        }
        void *send_r = scratch_send(kc);
        void *recv_r = scratch_recv(kc);
        for (int i = 0; i < comm->nranks; ++i) {
            std::memcpy(static_cast<char *>(send_r) + static_cast<size_t>(i) * chunk, p->sendbuff,
                        chunk);
        }
        if (kupl_shm_alltoall_init(send_r, static_cast<int>(chunk), KUPL_SHM_DATATYPE_CHAR, recv_r,
                                   static_cast<int>(chunk), KUPL_SHM_DATATYPE_CHAR, kc->comm,
                                   &kreq) != 0) {
            detail::err_record("kupl: allgather(alltoall) init FAILED");
            delete priv;
            return KuCCL_FAIL;
        }
        if (kupl_shm_request_start(kreq) != 0) {
            detail::err_record("kupl: allgather start FAILED");
            kupl_shm_request_free(kreq);
            delete priv;
            return KuCCL_FAIL;
        }
        priv->user_dst = p->recvbuff;
        priv->win_src = recv_r;
        priv->copy_bytes = total;
        break;
    }
    default:
        detail::err_record("kupl: unsupported coll op %d", static_cast<int>(op));
        delete priv;
        return KuCCL_FAIL;
    }

    priv->req = kreq;
    req->priv = priv;
    return KuCCL_SUCCESS;
}

/* ════════════════════════════════════════════════════════════
 * 请求推进：kupl start 同步驱动，到这里数据已就绪——拷回 + free
 * ════════════════════════════════════════════════════════════*/

static kucclResult_t kupl_req_test(kucclRequest *req, int *flag, kucclStatus_t *status) {
    auto *priv = static_cast<KuplReq *>(req->priv);
    if (priv == nullptr) {
        *flag = 1;
        *status = KuCCL_STATUS_ERROR;
        return KuCCL_SUCCESS;
    }
    if (priv->user_dst != nullptr && priv->copy_bytes > 0) {
        std::memcpy(priv->user_dst, priv->win_src, priv->copy_bytes);
    }
    if (priv->req != nullptr) {
        kupl_shm_request_free(priv->req);
    }
    delete priv;
    req->priv = nullptr;
    *flag = 1;
    *status = KuCCL_STATUS_COMPLETE;
    return KuCCL_SUCCESS;
}

static void kupl_req_cancel(kucclRequest *req) {
    auto *priv = static_cast<KuplReq *>(req->priv);
    if (priv == nullptr)
        return;
    /* kupl 无 cancel 语义：start 已同步返回，这里只回收 request */
    if (priv->req != nullptr)
        kupl_shm_request_free(priv->req);
    delete priv;
    req->priv = nullptr;
}

static void kupl_comm_purge(kucclComm * /*comm*/) {}

static kucclResult_t kupl_flush(kucclComm * /*comm*/) {
    /* shm 数据面同步完成，无在飞操作 */
    return KuCCL_SUCCESS;
}

const Ops *kupl_ops() {
    static const Ops ops = {
        "kupl",           &kupl_process_init, &kupl_process_finalize, &kupl_group_init,
        &kupl_group_fini, &kupl_coll_start,   &kupl_req_test,         &kupl_req_cancel,
        &kupl_comm_purge, &kupl_flush,
    };
    return &ops;
}

} /* namespace backend */
} /* namespace kuccl */

#else /* !KuCCL_HAVE_KUPL —— 存根（未找到 KUPL 时的占位实现） */

namespace kuccl {
namespace backend {

namespace {

kucclResult_t kupl_unavailable(const char *what) {
    detail::err_record("kupl backend: %s unavailable (built without KuCCL_HAVE_KUPL; "
                       "set KUPL_DIR to a KUPL install prefix and rebuild)",
                       what);
    return KuCCL_FAIL;
}

kucclResult_t kupl_process_init(kucclComm *comm) {
    KuCCL_LOG_ERROR(comm->rank, "KUCCL_BACKEND_TYPE=kupl: 本构建未链接 KUPL"
                                "（需 KUPL_DIR 指向 kupl 安装前缀后重新编译），"
                                "当前可用后端：xucg。");
    return kupl_unavailable("process_init");
}

void kupl_process_finalize() {}

kucclResult_t kupl_group_init(kucclComm *) {
    return kupl_unavailable("group_init");
}
void kupl_group_fini(kucclComm *) {}

kucclResult_t kupl_coll_start(kucclComm *, CollOp, const CollParams *, kucclRequest *) {
    return kupl_unavailable("collective start");
}

kucclResult_t kupl_req_test(kucclRequest *, int *, kucclStatus_t *) {
    return kupl_unavailable("request test");
}

void kupl_req_cancel(kucclRequest *) {}

void kupl_comm_purge(kucclComm *) {}

kucclResult_t kupl_flush(kucclComm *) {
    return kupl_unavailable("flush");
}

const Ops g_kupl_ops = {
    "kupl",           &kupl_process_init, &kupl_process_finalize, &kupl_group_init,
    &kupl_group_fini, &kupl_coll_start,   &kupl_req_test,         &kupl_req_cancel,
    &kupl_comm_purge, &kupl_flush,
};

} /* anonymous namespace */

const Ops *kupl_ops() {
    return &g_kupl_ops;
}

} /* namespace backend */
} /* namespace kuccl */

#endif /* KuCCL_HAVE_KUPL */
