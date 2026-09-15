/*
 * log.cpp —— kuccl 统一日志实现（见 log.hpp 的设计说明）
 */

#include "log.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>

namespace kuccl {
namespace detail {

LogLevel log_level() {
    static const LogLevel lv = [] {
        const char *e = std::getenv("KUCCL_LOG_LEVEL");
        if (e == nullptr || e[0] == '\0')
            return LogLevel::WARN;
        if (std::strcmp(e, "OFF") == 0)
            return LogLevel::OFF;
        if (std::strcmp(e, "ERROR") == 0)
            return LogLevel::ERROR;
        if (std::strcmp(e, "WARN") == 0)
            return LogLevel::WARN;
        if (std::strcmp(e, "INFO") == 0)
            return LogLevel::INFO;
        if (std::strcmp(e, "TRACE") == 0)
            return LogLevel::TRACE;
        /* 不认识的取值按 WARN 处理（保守静默） */
        return LogLevel::WARN;
    }();
    return lv;
}

static const char *level_tag(LogLevel lv) {
    switch (lv) {
    case LogLevel::ERROR:
        return "ERROR";
    case LogLevel::WARN:
        return "WARN";
    case LogLevel::INFO:
        return "INFO";
    case LogLevel::TRACE:
        return "TRACE";
    case LogLevel::OFF:
        break;
    }
    return "?";
}

void log_write(LogLevel lv, int rank, const char *file, int line, const char *fmt, ...) {
    /* 时间戳（秒级，排障足够） */
    char ts[16];
    {
        std::time_t t = std::time(nullptr);
        std::tm tm_v{};
        localtime_r(&t, &tm_v);
        std::strftime(ts, sizeof(ts), "%H:%M:%S", &tm_v);
    }

    std::fprintf(stderr, "[KUCCL][%s][%s][rank %d] ", ts, level_tag(lv), rank);

    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);

    /* 只保留文件名（全路径噪音大） */
    const char *base = file;
    for (const char *p = file; *p != '\0'; ++p) {
        if (*p == '/' || *p == '\\')
            base = p + 1;
    }
    std::fprintf(stderr, " (%s:%d)\n", base, line);
    std::fflush(stderr);
}

} /* namespace detail */
} /* namespace kuccl */
