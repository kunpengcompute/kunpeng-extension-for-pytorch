/*
 * log.cpp —— kuccl 统一日志实现（见 log.hpp 的设计说明）
 */

#include "log.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>

#include <chrono>
#include <unistd.h>
#include <vector>

namespace kuccl {
namespace detail {

/* 相对时间锚点：库加载时刻（steady_clock 单调，不受改系统时间影响）。
 * 日志里的 [+Ns] 即自此刻起算——多节点各进程锚点独立，用于单进程内看阶段。 */
static const std::chrono::steady_clock::time_point g_log_t0 = std::chrono::steady_clock::now();

/* ── 级别解析：大小写不敏感；数字 0-4；DEBUG=TRACE 别名 ── */
static LogLevel parse_level(const char *e)
{
    if (e == nullptr || e[0] == '\0')
        return LogLevel::WARN; /* 未设 = 默认 WARN */
    if (e[1] == '\0' && e[0] >= '0' && e[0] <= '4')
        return static_cast<LogLevel>(e[0] - '0');
    char up[8];
    std::size_t i = 0;
    for (; e[i] != '\0' && i < sizeof(up) - 1; ++i) {
        const char c = e[i];
        up[i] = (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c;
    }
    up[i] = '\0';
    if (std::strcmp(up, "OFF") == 0)
        return LogLevel::OFF;
    if (std::strcmp(up, "ERROR") == 0)
        return LogLevel::ERROR;
    if (std::strcmp(up, "WARN") == 0)
        return LogLevel::WARN;
    if (std::strcmp(up, "INFO") == 0)
        return LogLevel::INFO;
    if (std::strcmp(up, "TRACE") == 0 || std::strcmp(up, "DEBUG") == 0)
        return LogLevel::TRACE;
    /* 不认识的取值按 WARN 处理（保守静默） */
    return LogLevel::WARN;
}

LogLevel log_level()
{
    static const LogLevel lv = [] {
        return parse_level(std::getenv("KUCCL_LOG_LEVEL"));
    }();
    return lv;
}

static const char *level_tag(LogLevel lv)
{
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

/* ── rank 过滤：KUCCL_LOG_RANKS="0,3-5,17"（逗号分隔，支持区间；未设 = 全放行）。
 * 大 world 排障用：把 INFO/TRACE 限制到少数 rank，避免 128 进程刷屏。
 * 过滤对所有级别生效（含 ERROR）——想看全量就不设它。 */
static bool rank_allowed(int rank)
{
    struct Range {
        int lo, hi;
    };
    static const std::vector<Range> ranges = [] {
        std::vector<Range> v;
        const char *e = std::getenv("KUCCL_LOG_RANKS");
        if (e == nullptr || e[0] == '\0')
            return v;
        int bad = 0;
        const char *p = e;
        while (*p != '\0') {
            while (*p == ',' || *p == ' ')
                ++p;
            if (*p == '\0')
                break;
            char *end = nullptr;
            const long lo = std::strtol(p, &end, 10);
            if (end == p) {
                ++bad;
                ++p;
                continue;
            }
            p = end;
            long hi = lo;
            if (*p == '-') {
                ++p;
                hi = std::strtol(p, &end, 10);
                if (end == p) { /* 尾随破折号（"3-"）按无效片段处理 */
                    ++bad;
                    continue;
                }
                p = end;
            }
            if (lo >= 0 && hi >= lo && hi < (1L << 20))
                v.push_back({static_cast<int>(lo), static_cast<int>(hi)});
            else
                ++bad;
        }
        if (v.empty())
            std::fprintf(stderr, "[KuCCL][WARN] KUCCL_LOG_RANKS='%s' 无有效 rank，按不过滤处理\n", e);
        else if (bad > 0)
            std::fprintf(stderr, "[KuCCL][WARN] KUCCL_LOG_RANKS='%s' 有 %d 个无效片段已跳过\n", e, bad);
        return v;
    }();
    if (ranges.empty() || rank < 0)
        return true; /* 未设过滤 / 无 rank 上下文的日志不过滤 */
    for (const auto &r : ranges)
        if (rank >= r.lo && rank <= r.hi)
            return true;
    return false;
}

/* ── host 名：进程内取一次，超长截断（多节点聚合日志时区分来源机器）── */
static const char *log_host()
{
    static const char *cached = [] {
        static char buf[24];
        if (::gethostname(buf, sizeof(buf) - 1) != 0) {
            std::memcpy(buf, "unknown", sizeof("unknown"));
            return buf;
        }
        buf[16] = '\0';
        return buf;
    }();
    return cached;
}

/* ── 截断安全的追加拼装：返回追加后的已用长度 ── */
static int append_line(char *buf, std::size_t cap, int off, const char *fmt, ...)
{
    if (off < 0 || static_cast<std::size_t>(off) + 1 >= cap)
        return off;
    va_list ap;
    va_start(ap, fmt);
    const int n = std::vsnprintf(buf + off, cap - static_cast<std::size_t>(off), fmt, ap);
    va_end(ap);
    if (n <= 0)
        return off;
    const int room = static_cast<int>(cap - static_cast<std::size_t>(off)) - 1;
    return off + (n < room ? n : room);
}

void log_write(LogLevel lv, int rank, const char *file, int line, const char *fmt, ...)
{
    if (!rank_allowed(rank))
        return;

    /* 墙钟（毫秒级）+ 相对库加载的流逝秒 */
    char ts[20];
    double elapsed_s;
    {
        struct timespec now;
        clock_gettime(CLOCK_REALTIME, &now);
        std::tm tm_v{};
        localtime_r(&now.tv_sec, &tm_v);
        std::strftime(ts, sizeof(ts), "%H:%M:%S", &tm_v);
        std::snprintf(ts + 8, sizeof(ts) - 8, ".%03d", static_cast<int>(now.tv_nsec / 1000000));
        elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_log_t0).count();
    }

    /* 整行单缓冲拼装后一次写出：多线程不交错（glibc 对 stderr 的单次写带锁） */
    char buf[1024];
    int n = append_line(buf, sizeof(buf), 0, "[KuCCL][%s][%s][+%.1fs][%s][rank %d] ", level_tag(lv), ts, elapsed_s,
                        log_host(), rank);

    /* 正文（变参）直接 vsnprintf 到剩余空间 */
    va_list ap;
    va_start(ap, fmt);
    if (n >= 0 && static_cast<std::size_t>(n) + 1 < sizeof(buf)) {
        const int m = std::vsnprintf(buf + n, sizeof(buf) - static_cast<std::size_t>(n), fmt, ap);
        if (m > 0) {
            const int room = static_cast<int>(sizeof(buf) - static_cast<std::size_t>(n)) - 1;
            n += (m < room ? m : room);
        }
    }
    va_end(ap);

    /* 只保留文件名（全路径噪音大） */
    const char *base = file;
    for (const char *p = file; *p != '\0'; ++p) {
        if (*p == '/' || *p == '\\')
            base = p + 1;
    }
    n = append_line(buf, sizeof(buf), n, " (%s:%d)\n", base, line);

    /* 极端截断兜底：行尾必须有换行 */
    const std::size_t len = std::strlen(buf);
    if (len == 0 || buf[len - 1] != '\n') {
        if (len + 1 < sizeof(buf)) {
            buf[len] = '\n';
            buf[len + 1] = '\0';
        } else {
            buf[sizeof(buf) - 2] = '\n';
            buf[sizeof(buf) - 1] = '\0';
        }
    }

    std::fputs(buf, stderr);
    std::fflush(stderr);
}

} /* namespace detail */
} /* namespace kuccl */
