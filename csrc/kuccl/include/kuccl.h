#ifndef KuCCL_H_
#define KuCCL_H_

/*
 * kuccl 对外 C 接口
 *
 * 本头文件属于 kuccl 纯通信库：
 *   - 纯 C 接口，不依赖 PyTorch / MPI / Python
 *   - 任何 C/C++ 程序可直接 #include "kuccl.h" 并链接 libkuccl.so
 *
 * 对外集合通信接口 4 个：
 *   kucclAllGather / kucclAllReduce / kucclBroadcast / kucclAllToAll
 *
 * 全部接口（12 个公开接口）：
 *   通信域管理：kucclGetUniqueId / kucclCommInitRankConfig /
 *              kucclCommDestroy / kucclCommSplit
 *   集合通信：  kucclAllGather / kucclAllReduce / kucclBroadcast / kucclAllToAll
 *   异步测试：  kucclTest
 *   辅助设施：kucclGetLastError / kucclCommAbort / kucclFlushComm
 *
 * 后端选择：kucclConfig_t.backend（"xucg" 默认 / "kupl" 预留），
 *           或环境变量 KUCCL_BACKEND_TYPE（详见 types.hpp）。
 *
 * MPI 直调完整示例（可编译、可直接跑）：examples/coll_test_mpi.cpp
 * PyTorch 接入：import kuccl（详见 torch_kuccl/README.md）
 */

#include <stddef.h>
#include "types.hpp"

/* ════════════════════════════════════════════════════════════
 * 符号可见性（KuCCL_API 宏模式）
 *
 * 库编译时带 -fvisibility=hidden + -DKuCCL_EXPORTS：
 *   仅公开接口可见，内部实现全部隐藏（更小的符号表、防误用）。
 * 用户侧 #include 本头文件时 KuCCL_EXPORTS 未定义 → 宏为空，不影响使用。
 * ════════════════════════════════════════════════════════════*/
#if defined(KuCCL_EXPORTS)
#define KuCCL_API __attribute__((visibility("default")))
#else
#define KuCCL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ════════════════════════════════════════════════════════════
 * 通信域管理
 * ════════════════════════════════════════════════════════════*/

/* 生成全局唯一 ID，用于 kucclCommInitRankConfig 创建通信域。
 * 通常 rank 0 调用，然后通过 MPI_Bcast 或其他方式广播给所有进程。 */
KuCCL_API kucclResult_t kucclGetUniqueId(kucclUniqueId *uniqueId);

/* 创建 kuccl 通信域。
 *   nranks  - 通信域总进程数
 *   commId  - kucclGetUniqueId 生成的 ID（所有进程必须相同）
 *   rank    - 当前进程 ID
 *   config  - OOB 回调配置（oob_allgather 必填，oob_ctx 原样传回回调；
 *             backend 可选后端名，NULL = 默认 xucg） */
KuCCL_API kucclResult_t kucclCommInitRankConfig(kucclComm_t *comm, int nranks, kucclUniqueId commId,
                                                int rank, kucclConfig_t *config);

/* 销毁 kuccl 通信域。 */
KuCCL_API kucclResult_t kucclCommDestroy(kucclComm_t comm);

/* 通信域切分（语义同 MPI_Comm_split）。
 *   color   - 分组标签，同 color 的进程进入同一子组（color<0 表示不加入任何子组）
 *   key     - 子组内排序
 *   config  - 额外配置，可为 NULL（从父通信域继承） */
KuCCL_API kucclResult_t kucclCommSplit(kucclComm_t comm, int color, int key, kucclComm_t *newcomm,
                                       kucclConfig_t *config);

/* ════════════════════════════════════════════════════════════
 * 错误信息 + 安全终止
 *
 * ════════════════════════════════════════════════════════════*/

/* 返回本线程最近一次错误的详文并清空。
 *   - nullptr = 无错误记录
 *   - 返回指针在下一次本线程调用本接口前有效
 *   - 线程安全（TLS 存储）
 *   - 典型用法：
 *       if (kucclAllReduce(...) != KuCCL_SUCCESS)
 *           fprintf(stderr, "%s\n", kucclGetLastError()); */
KuCCL_API const char *kucclGetLastError(void);

/* 安全终止通信域。
 *   - 立即标记 comm 为 aborted：后续所有算子调用返回 KuCCL_FAIL
 *     （错误信息注明 aborted），不再触碰底层通信资源
 *   - 不保证在飞请求的数据完整性（用于故障恢复，
 *     不能当正常销毁用——正常销毁仍用 kucclCommDestroy）
 *   - 多次调用幂等；线程安全 */
KuCCL_API kucclResult_t kucclCommAbort(kucclComm_t comm);

/* Flush 通信域底层传输（阻塞到该 comm 所有在飞操作落地）。
 * 阻塞到该 comm 所有在飞操作落地。 */
KuCCL_API kucclResult_t kucclFlushComm(kucclComm_t comm);

/* ════════════════════════════════════════════════════════════
 * 集合通信操作（异步：启动后立即返回，用 kucclTest 查询完成）
 * ════════════════════════════════════════════════════════════*/

/* AllGather：每个进程贡献 sendcount 个元素，最终所有进程收到全部拼接。
 *   recvbuff 需 nranks * sendcount 个元素。
 *
 * ★ count 参数语义（全库统一约定）：
 *   sendcount 始终是【元素个数】，不是字节数。
 *   内部实现会转换为字节（sendcount × sizeof(datatype)）再传给后端，
 *   用户不需要关心这个转换。
 *
 * ★ 实现说明（count 语义全库统一为元素个数）：
 *   AllReduce  : 真实 dtype + 元素数（归约必须知道元素类型）
 *   其余算子   : 默认内部转 UINT8 + 字节数传后端（KUCCL_VDT=elem
 *                可切元素语义）。两种语义在 UCG 侧的 plan 选择不同。 */
KuCCL_API kucclResult_t kucclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount,
                                       kucclDataType_t datatype, kucclComm_t comm,
                                       kucclRequest_t *request);

/* AllReduce：所有进程的数据按 reduction_op 归约，结果写入 recvbuff。
 *   recvbuff 可与 sendbuff 相同（原地归约）。
 *
 *   count 是元素个数（不是字节数）。
 *   内部必须用真实 dtype（不能用 UINT8），否则逐字节归约出错。 */
KuCCL_API kucclResult_t kucclAllReduce(void *sendbuff, void *recvbuff, size_t count,
                                       kucclDataType_t datatype, kucclRedOp_t reduction_op,
                                       kucclComm_t comm, kucclRequest_t *request);

/* Broadcast：root 进程的数据广播给所有进程。
 *   root 的 sendbuff 为数据源；recvbuff 为收发缓冲（root 处先拷贝） */
KuCCL_API kucclResult_t kucclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                                       kucclDataType_t datatype, int root, kucclComm_t comm,
                                       kucclRequest_t *request);

/* AllToAll：所有进程之间等长全交换。
 *   sendbuff/recvbuff 需 nranks * count 个元素 */
KuCCL_API kucclResult_t kucclAllToAll(const void *sendbuff, void *recvbuff, size_t count,
                                      kucclDataType_t datatype, kucclComm_t comm,
                                      kucclRequest_t *request);

/* ════════════════════════════════════════════════════════════
 * 异步请求测试
 * ════════════════════════════════════════════════════════════*/

/* 查询 request 是否完成。
 *   request - 要查询的请求（完成后自动置 NULL 并释放）
 *   flag    - 输出：1=完成，0=未完成
 *   status  - 输出：KuCCL_STATUS_COMPLETE / KuCCL_STATUS_ERROR
 *
 *   KUCCL_TEST_TIMEOUT_MS=N（默认 0=无限等待）配置看门狗：
 *   超时返回 KuCCL_FAIL 并清理请求（防止对端掉线导致永久阻塞）。 */
KuCCL_API kucclResult_t kucclTest(kucclRequest_t *request, int *flag, kucclStatus_t *status);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* KuCCL_H_ */
