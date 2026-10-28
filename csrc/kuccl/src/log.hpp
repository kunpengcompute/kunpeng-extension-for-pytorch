#ifndef KuCCL_LOG_HPP_
#define KuCCL_LOG_HPP_

/*
 * kuccl 统一日志
 *
 * 级别由 KUCCL_LOG_LEVEL 控制（OFF/ERROR/WARN/INFO/TRACE，默认 WARN）：
 *   TRACE 逐调用细节（仅排障）   INFO 关键里程碑
 *   WARN  可继续但有隐患         ERROR 失败路径
 * 落盘：KUCCL_LOG_FILE=path 在 stderr 之外同步写文件；"%r" 按 rank 分文件。
 * 过滤：KUCCL_LOG_RANKS="0,3-5" 只放行列出的 rank（对所有级别生效）。
 * 输出为 stderr 单行（不污染 stdout 解析，整行一次写出）：
 *   [KuCCL][LEVEL][HH:MM:SS.mmm][+123.4s][host][rank R] message (file:line)
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

#define KuCCL_LOG_ERROR(rank, ...)                                                              \
    do {                                                                                        \
        using kuccl::detail::LogLevel;                                                          \
        if (kuccl::detail::log_level() >= LogLevel::ERROR)                                      \
            kuccl::detail::log_write(LogLevel::ERROR, (rank), __FILE__, __LINE__, __VA_ARGS__); \
    } while (0)

#define KuCCL_LOG_WARN(rank, ...)                                                              \
    do {                                                                                       \
        using kuccl::detail::LogLevel;                                                         \
        if (kuccl::detail::log_level() >= LogLevel::WARN)                                      \
            kuccl::detail::log_write(LogLevel::WARN, (rank), __FILE__, __LINE__, __VA_ARGS__); \
    } while (0)

#define KuCCL_LOG_INFO(rank, ...)                                                              \
    do {                                                                                       \
        using kuccl::detail::LogLevel;                                                         \
        if (kuccl::detail::log_level() >= LogLevel::INFO)                                      \
            kuccl::detail::log_write(LogLevel::INFO, (rank), __FILE__, __LINE__, __VA_ARGS__); \
    } while (0)

#define KuCCL_LOG_TRACE(rank, ...)                                                              \
    do {                                                                                        \
        using kuccl::detail::LogLevel;                                                          \
        if (kuccl::detail::log_level() >= LogLevel::TRACE)                                      \
            kuccl::detail::log_write(LogLevel::TRACE, (rank), __FILE__, __LINE__, __VA_ARGS__); \
    } while (0)

#endif /* KuCCL_LOG_HPP_ */
