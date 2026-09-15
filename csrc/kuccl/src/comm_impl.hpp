#ifndef KUCCL_COMM_IMPL_HPP_
#define KUCCL_COMM_IMPL_HPP_

/*
 * kuccl 通信域内部实体（libkuccl 私有，不安装、不对用户暴露）
 *
 * kucclComm 是后端无关的通信域描述：只承载组信息、OOB 配置与
 * 后端选择结果，底层资源（UCP/UCG 上下文、组句柄、rank 映射等）
 * 全部由后端在 be_ctx 里自行管理（见 backend/backend.hpp）。
 *
 * 核心机制（均由 xucg 后端承接，见 backend/xucg.cpp）：
 *   - 进程级单例 + 全局地址表
 *   - 两阶段变长地址 allgather
 *   - FNV-1a group_id 哈希 + instance_token 防撞
 *   - 3 个 UCG 回调（oob_cb / get_ucp_ep / get_ucp_worker/context）
 */

#include "types.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus

namespace kuccl {
namespace backend {

/* 后端操作表（定义见 backend/backend.hpp） */
struct Ops;

} /* namespace backend */
} /* namespace kuccl */

extern "C" {

/* ── 异步请求 ──────────────────────────────────────────────
 * 后端无关的外壳：状态 + 看门狗起点 + 后端私有状态指针。
 * 后端在 priv 里管理自己的请求句柄（xucg: ucg_request_h 等）。 */
struct kucclRequest {
    kucclStatus_t status = KUCCL_STATUS_PENDING;
    /* 启动时刻：看门狗计时与剖析共用 */
    std::chrono::steady_clock::time_point t0{};
    /* 算子名（看门狗超时信息用，指向静态字符串） */
    const char *op_name = "op";
    /* 所属后端（kucclTest 推进/看门狗取消时回调） */
    const kuccl::backend::Ops *be = nullptr;
    /* 后端私有状态（由后端分配与释放；kucclTest 完成路径回调后端回收） */
    void *priv = nullptr;
};

/* ── 通信域结构体 ────────────────────────────────────────── */
struct kucclComm {
    int rank;   /* 组内 rank */
    int nranks; /* 总进程数 */

    /* ---- OOB 配置（用户注入的回调，后端引导期用）──── */
    kucclConfig_t oob_config; /* 回调函数指针 + 用户上下文 + 后端名 */
    int has_oob;

    /* ---- abort 状态（kucclCommAbort 置位）────
     * atomic：abort 可能从任意线程调（信号处理/看门狗）。
     * 算子入口快速检查——aborted 后拒绝一切新操作。 */
    std::atomic<int> aborted{0};

    /* ---- 后端（kucclCommInitRankConfig 时解析并固定）──── */
    const kuccl::backend::Ops *be = nullptr;
    void *be_ctx = nullptr; /* 后端私有状态（xucg: 组句柄/地址映射等） */
};

} /* extern "C" */

namespace kuccl {

/* 进程级终结（释放后端进程级资源；进程退出/测试收尾时调用） */
void finalize();

namespace detail {

/* ── API 串行化锁（定义在 lock.cpp）──────────────────────
 * 后端以 SINGLE 线程模式创建底层资源，全部公开接口在入口
 * 用这把递归锁做外部串行化（recursive：CommSplit 内部会
 * 调 kucclCommInitRankConfig，重入）。 */
extern std::recursive_mutex g_api_lock;

/* 错误信息记录（定义在 error.cpp，TLS 存储）——失败路径调用 */
void err_record(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
const char *err_peek_last();
void err_append(const char *ctx);

/* OOB allgather 快捷封装：走 config.oob_allgather 回调（返回 0 = 成功）。
 * 定义在 comm.cpp；后端引导期与 CommSplit 共用。 */
int oob_exchange(const kucclConfig_t *cfg, const void *sendbuf, void *recvbuf, int count);

/* 当前活跃后端（首个成功建组的后端；finalize 用） */
void set_active_backend(const backend::Ops *be);
const backend::Ops *active_backend();

} /* namespace detail */
} /* namespace kuccl */

#endif /* __cplusplus */

#endif /* KUCCL_COMM_IMPL_HPP_ */
