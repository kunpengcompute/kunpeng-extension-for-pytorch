#ifndef KuCCL_LOG_HPP_
#define KuCCL_LOG_HPP_

/*
 * kuccl 统一日志（四级分层）
 *
 * 四个级别（TRACE/INFO/WARN/ERROR）：
 *   KuCCL_LOG_TRACE - 逐调用细节（地址交换、dt 缓存命中……只用于问题定位）
 *   KuCCL_LOG_INFO  - 关键里程碑（进程初始化完成、建组成功、拓扑结果）
 *   KuCCL_LOG_WARN  - 可继续但有隐患（回退路径、资源复用失败降级）
 *   KuCCL_LOG_ERROR - 失败路径（初始化失败、算子报错）
 *
 * 级别控制：环境变量 KUCCL_LOG_LEVEL，取值 OFF/ERROR/WARN/INFO/TRACE
 *           （DEBUG=TRACE 别名；也接受数字 0-4；大小写不敏感），
 *           默认 WARN——生产静默，只出警告和错误；排障时调到 INFO/TRACE。
 *
 * rank 过滤：环境变量 KUCCL_LOG_RANKS（"0,3-5,17"——逗号分隔支持区间，
 *           只放行列出的 rank；未设 = 全放行）。大 world 排障时把
 *           INFO/TRACE 限制到少数 rank，避免上百进程刷屏。
 *           注意过滤对所有级别生效（含 ERROR）。
 *
 * 输出：stderr（不污染 benchmark 的 stdout 解析），单行格式：
 *       [KuCCL][LEVEL][HH:MM:SS.mmm][+123.4s][host][rank R] message (file:line)
 *       - 毫秒墙钟 + 相对库加载的流逝秒（steady_clock，看阶段用）
 *       - host：多节点日志聚合到一处时区分来源机器
 *       - 整行一次写出（多线程不交错）
 */

#include "types.hpp"

namespace kuccl {
namespace detail {

enum class LogLevel : int {
    OFF = 0, /* 全关 */
    ERROR = 1,
    WARN = 2,
    INFO = 3,
    TRACE = 4,
};

/* 当前生效级别（读 KUCCL_LOG_LEVEL，进程内只读一次） */
LogLevel log_level();

/* 日志写入（内部实现，业务代码用下面的宏） */
void log_write(LogLevel lv, int rank, const char *file, int line, const char *fmt, ...)
    __attribute__((format(printf, 5, 6)));

} /* namespace detail */
} /* namespace kuccl */

#define KuCCL_LOG_ERROR(rank, ...)                                                                 \
    do {                                                                                           \
        using kuccl::detail::LogLevel;                                                             \
        if (kuccl::detail::log_level() >= LogLevel::ERROR)                                         \
            kuccl::detail::log_write(LogLevel::ERROR, (rank), __FILE__, __LINE__, __VA_ARGS__);    \
    } while (0)

#define KuCCL_LOG_WARN(rank, ...)                                                                  \
    do {                                                                                           \
        using kuccl::detail::LogLevel;                                                             \
        if (kuccl::detail::log_level() >= LogLevel::WARN)                                          \
            kuccl::detail::log_write(LogLevel::WARN, (rank), __FILE__, __LINE__, __VA_ARGS__);     \
    } while (0)

#define KuCCL_LOG_INFO(rank, ...)                                                                  \
    do {                                                                                           \
        using kuccl::detail::LogLevel;                                                             \
        if (kuccl::detail::log_level() >= LogLevel::INFO)                                          \
            kuccl::detail::log_write(LogLevel::INFO, (rank), __FILE__, __LINE__, __VA_ARGS__);     \
    } while (0)

#define KuCCL_LOG_TRACE(rank, ...)                                                                 \
    do {                                                                                           \
        using kuccl::detail::LogLevel;                                                             \
        if (kuccl::detail::log_level() >= LogLevel::TRACE)                                         \
            kuccl::detail::log_write(LogLevel::TRACE, (rank), __FILE__, __LINE__, __VA_ARGS__);    \
    } while (0)

#endif /* KuCCL_LOG_HPP_ */
