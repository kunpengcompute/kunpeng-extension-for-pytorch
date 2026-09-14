#ifndef KUCCL_LOG_HPP_
#define KUCCL_LOG_HPP_

/*
 * kuccl 统一日志（四级分层）
 *
 * 四个级别（TRACE/INFO/WARN/ERROR）：
 *   KUCCL_LOG_TRACE - 逐调用细节（地址交换、dt 缓存命中……只用于问题定位）
 *   KUCCL_LOG_INFO  - 关键里程碑（进程初始化完成、建组成功）
 *   KUCCL_LOG_WARN  - 可继续但有隐患（回退路径、资源复用失败降级）
 *   KUCCL_LOG_ERROR - 失败路径（初始化失败、算子报错）
 *
 * 级别控制：环境变量 KUCCL_LOG_LEVEL（OFF/ERROR/WARN/INFO/TRACE），
 *           默认 WARN——生产静默，只出警告和错误；排障时调到 INFO/TRACE。
 *
 * 输出：stderr（不污染 benchmark 的 stdout 解析），
 *       单行格式：[KUCCL][LEVEL][时间][rank R] message (file:line)
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

#define KUCCL_LOG_ERROR(rank, ...)                                                                 \
    do {                                                                                           \
        using kuccl::detail::LogLevel;                                                             \
        if (kuccl::detail::log_level() >= LogLevel::ERROR)                                         \
            kuccl::detail::log_write(LogLevel::ERROR, (rank), __FILE__, __LINE__, __VA_ARGS__);    \
    } while (0)

#define KUCCL_LOG_WARN(rank, ...)                                                                  \
    do {                                                                                           \
        using kuccl::detail::LogLevel;                                                             \
        if (kuccl::detail::log_level() >= LogLevel::WARN)                                          \
            kuccl::detail::log_write(LogLevel::WARN, (rank), __FILE__, __LINE__, __VA_ARGS__);     \
    } while (0)

#define KUCCL_LOG_INFO(rank, ...)                                                                  \
    do {                                                                                           \
        using kuccl::detail::LogLevel;                                                             \
        if (kuccl::detail::log_level() >= LogLevel::INFO)                                          \
            kuccl::detail::log_write(LogLevel::INFO, (rank), __FILE__, __LINE__, __VA_ARGS__);     \
    } while (0)

#define KUCCL_LOG_TRACE(rank, ...)                                                                 \
    do {                                                                                           \
        using kuccl::detail::LogLevel;                                                             \
        if (kuccl::detail::log_level() >= LogLevel::TRACE)                                         \
            kuccl::detail::log_write(LogLevel::TRACE, (rank), __FILE__, __LINE__, __VA_ARGS__);    \
    } while (0)

#endif /* KUCCL_LOG_HPP_ */
