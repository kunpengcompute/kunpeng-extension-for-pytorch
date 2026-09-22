#ifndef KuCCL_BACKEND_HPP_
#define KuCCL_BACKEND_HPP_

/*
 * kuccl 后端接口（双后端架构）
 *
 * kuccl 对上只暴露 kuccl.h 的稳定 C API；对下由可插拔后端承接：
 *   ┌────────────────────────────────────────────┐
 *   │  kuccl.h（4 集合通信 + 通信域管理）         │
 *   ├────────────────────────────────────────────┤
 *   │  coll.cpp / comm.cpp（校验/锁/分发）        │
 *   ├────────────────────┬───────────────────────┤
 *   │  xucg 后端          │  kupl 后端             │
 *   │  UCG 公有接口       │  KUPL shm（单机快路径） │
 *   └────────────────────┴───────────────────────┘
 *
 * 后端边界：
 *   - 集合请求的全部状态判定与推进只走 UCG 公有接口
 *     （ucg_request_*_init / ucg_request_start / ucg_request_test /
 *      ucg_progress），错误码一律 UCG_INPROGRESS / UCG_OK，
 *      禁止在集合路径直接判 UCS_INPROGRESS 等 UCX 内部状态；
 *   - UCP 仅作为 xucg 后端的私有资源层（ucg 的编程模型要求
 *     调用方注入 ucp context/worker/ep——见 ucg_global_init 的
 *     oob_resource 回调），相关调用全部封闭在本后端文件内。
 *
 * 注册方式：各后端实现一个 `const Ops* kuccl_backend_<name>()`，
 * backend::get(name) 做名字分发（"ucg" 是 "xucg" 的兼容别名）。
 */

#include "../comm_impl.hpp"
#include "../log.hpp"

namespace kuccl {
namespace backend {

/* 集合算子类型（后端内部的分发用，与公开 API 一一对应） */
enum class CollOp : int {
    AllGather = 0,
    AllReduce = 1,
    Broadcast = 2,
    AllToAll = 3,
};

/* 集合调用参数（一次调用的全部数据面信息） */
struct CollParams {
    const void *sendbuff = nullptr;
    void *recvbuff = nullptr;
    /* 通用计数：AllGather=sendcount；AllReduce/Broadcast/AllToAll=count
     * （元素个数，不是字节数；字节换算由后端按 AllGather 约定处理） */
    size_t count = 0;
    kucclDataType_t datatype = kucclUint8;
    kucclRedOp_t redop = kucclSum;
    int root = 0;
};

/* 后端操作表——kuccl 骨架（coll.cpp/comm.cpp）只通过它触碰底层 */
struct Ops {
    const char *name;

    /* 进程级一次性资源（UCP/UCG 上下文、地址交换等）。首次建组时调用，
     * 后端内部保证幂等。失败返回 KuCCL_FAIL 并已记录错误信息。 */
    kucclResult_t (*process_init)(kucclComm *comm);

    /* 进程级终结（kuccl::finalize 调用，进程退出前） */
    void (*process_finalize)();

    /* 通信组建组：bootstrap 交换 + rank 映射 + 底层组创建。
     * 成功后 comm->be_ctx 由后端持有。 */
    kucclResult_t (*group_init)(kucclComm *comm);

    /* 通信域销毁（回收 comm->be_ctx；调用方已持锁且已 purge） */
    void (*group_fini)(kucclComm *comm);

    /* 启动一次集合操作：填充 req->priv，异步语义（立即返回） */
    kucclResult_t (*coll_start)(kucclComm *comm, CollOp op, const CollParams *p, kucclRequest *req);

    /* 推进请求：未完成 flag=0；完成/出错 flag=1 并回收 req->priv
     * （由 kucclTest 调用，调用方已持锁；watchdog 在外层） */
    kucclResult_t (*req_test)(kucclRequest *req, int *flag, kucclStatus_t *status);

    /* 取消请求（看门狗超时路径；UCG cancel 语义：回收在飞请求与 priv） */
    void (*req_cancel)(kucclRequest *req);

    /* 清理该 comm 关联的后端私有状态（abort / destroy 前调用）：
     * 持久化请求池、在飞请求的尽力回收等 */
    void (*comm_purge)(kucclComm *comm);

    /* flush：阻塞到该 comm 在飞操作落地 */
    kucclResult_t (*flush)(kucclComm *comm);
};

/* 按名取后端。"xucg"（默认，"ucg" 为别名）/ "kupl"（预留存根）。
 * 未知名字返回 nullptr。 */
const Ops *get(const char *name);

/* 各后端注册点（实现在各自 .cpp） */
const Ops *xucg_ops();
const Ops *kupl_ops();

} /* namespace backend */
} /* namespace kuccl */

#endif /* KuCCL_BACKEND_HPP_ */
