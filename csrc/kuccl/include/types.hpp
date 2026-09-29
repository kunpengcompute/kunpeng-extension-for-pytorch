#ifndef KuCCL_TYPES_HPP_
#define KuCCL_TYPES_HPP_

/*
 * KuCCL 公共类型定义
 *
 * 本文件属于 kuccl 纯库，禁止依赖任何框架头文件
 * （不 include torch/mpi/pybind —— 那些属于 kuccl_backend 适配层）。
 *
 * 类型命名：kucclResult_t / kucclComm_t / kucclUniqueId / kucclDataType_t 等
 * 保持 C 兼容（extern "C"）。
 */

#include <stddef.h>
#include <stdint.h>

/* ════════════════════════════════════════════════════════════
 * 符号可见性宏（KuCCL_API）
 *
 * 放在 types.hpp（所有 cpp 都会 include 它），保证库内各文件统一。
 * 编译库时 -DKuCCL_EXPORTS → 符号导出；用户 include 时宏为空。
 * ════════════════════════════════════════════════════════════*/
#if defined(KuCCL_EXPORTS)
#define KuCCL_API __attribute__((visibility("default")))
#else
#define KuCCL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── 返回码 ────────────────────────────────────────────── */
/* 结果码：0 成功；-1 失败。
 * 失败的具体原因走两条文本通道：
 *   kucclGetLastError()（kuccl/UCG 两层的报错原文，含 UCG st 码）
 *   KUCCL_LOG_* 日志行                                                  */
typedef enum {
    KuCCL_SUCCESS = 0,
    KuCCL_FAIL = -1
} kucclResult_t;

/* ── 通信域状态（kucclTest 返回）──────────────────────── */
typedef enum {
    KuCCL_STATUS_PENDING = 0,
    KuCCL_STATUS_COMPLETE = 1,
    KuCCL_STATUS_ERROR = -1
} kucclStatus_t;

/* ── 全局唯一 ID ──────────────────────────────────────────
 * 生成全局唯一 ID
 * 用于 kucclCommInitRankConfig 创建通信域 */
typedef struct kucclUniqueId {
    char data[128];
} kucclUniqueId;

/* ── 通信域（不透明指针）───────────────────────────────── */
typedef struct kucclComm kucclComm;
typedef kucclComm *kucclComm_t;

/* ── 异步请求（不透明指针）──────────────────────────────── */
typedef struct kucclRequest kucclRequest;
typedef kucclRequest *kucclRequest_t;

/* ── 数据类型 ───────────────────────────────────────────── */
typedef enum {
    kucclInt8 = 0,
    kucclUint8 = 1,
    kucclInt32 = 2,
    kucclUint32 = 3,
    kucclInt64 = 4,
    kucclUint64 = 5,
    kucclFloat = 6,
    kucclDouble = 7
} kucclDataType_t;

/* ── 归约操作 ───────────────────────────────────────────── */
typedef enum {
    kucclSum = 0,
    kucclMin = 1,
    kucclMax = 2
} kucclRedOp_t;

/* ════════════════════════════════════════════════════════════
 * OOB 回调签名 —— kuccl 库唯一的引导通道（留给调用方注入）
 *
 * kuccl 库本身不做 TCPStore/MPI，引导期的数据交换（地址、bootstrap 记录、
 * CommSplit 的 color）全部通过用户注入的这两个回调完成：
 *   - MPI 程序：     config.oob_allgather = MPI_Allgather
 *   - PyTorch 程序： kuccl_backend 适配层注入基于 c10d::Store 的封装
 * ════════════════════════════════════════════════════════════*/

/* OOB allgather：语义对齐 MPI_Allgather
 *   sendbuf     - 本进程贡献的数据
 *   recvbuf     - 收齐后按 rank 拼接（nranks * count 字节）
 *   count       - 每进程贡献的字节数
 *   user_ctx    - 用户上下文（config.oob_ctx 原样传回，如 MPI_Comm*）
 * 返回 0 表示成功（对齐 MPI 的 int 返回约定） */
typedef int (*kucclOobAllgatherFn)(const void *sendbuf, void *recvbuf, int count, void *user_ctx);

/* OOB broadcast：语义对齐 MPI_Bcast（config.oob_bcast，供扩展使用）
 *   buffer   - 广播缓冲区（root 为数据源）
 *   count    - 字节数
 *   root     - 广播根
 *   user_ctx - 用户上下文
 * 返回 0 表示成功 */
typedef int (*kucclOobBcastFn)(void *buffer, int count, int root, void *user_ctx);

/* ── 配置（OOB 回调函数指针 + 用户上下文 + 后端选择）─────────
 */
typedef struct kucclConfig {
    kucclOobAllgatherFn oob_allgather; /* 必填：引导期数据交换 */
    kucclOobBcastFn oob_bcast;         /* 选填：广播（CommSplit/UniqueId 分发可用） */
    void *oob_ctx;                     /* 用户上下文，回调时原样传回 */
    /* 后端选择：
     *   nullptr / "xucg" - 默认，UCG 集合通信后端（单机 + 跨节点）
     *   "ucg"            - "xucg" 的兼容别名
     *   "kupl"           - 预留（单机共享内存优化路径，尚未实现）
     * 环境变量 KUCCL_BACKEND_TYPE 可覆盖（config 优先级更高） */
    const char *backend;
} kucclConfig_t;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* KuCCL_TYPES_HPP_ */
