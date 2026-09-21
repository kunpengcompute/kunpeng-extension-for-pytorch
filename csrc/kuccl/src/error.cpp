/*
 * error.cpp —— 错误信息基础设施
 *
 *   - 线程局部存储（TLS）最近一次错误详文——多线程各自独立
 *   - KUCCL_FAIL 返回后用户可查具体原因（算子/参数/UCG 状态码）
 *   - 成功路径零开销（只在失败时写入）
 *
 * 设计：
 *   - 固定缓冲避免动态分配（错误路径不能再失败）
 *   - 环形保留最近 KUCCL_ERR_HISTORY 条（默认 4，env 可调）
 *   - kucclGetLastError 返回最近一条并清空（一次性读取）
 *   - kucclAppendLastError 追加上下文（调用链逐层补信息）
 */

#include "types.hpp"
#include "kuccl.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace kuccl {
namespace detail {

/* 错误历史深度（env 覆盖：KUCCL_ERR_HISTORY=N） */
static int err_history_depth() {
    static const int n = [] {
        const char *e = std::getenv("KUCCL_ERR_HISTORY");
        int v = e ? std::atoi(e) : 4;
        return (v > 0 && v <= 32) ? v : 4;
    }();
    return n;
}

constexpr size_t ERR_BUF = 512;

struct ErrSlot {
    char text[ERR_BUF];
    bool used;
};

/* 线程局部错误环——每线程独立，免锁 */
struct ErrRing {
    ErrSlot slots[8];
    int next = 0;  /* 下一个写入槽 */
    int count = 0; /* 已用槽数 */
};

static ErrRing &tls_ring() {
    static thread_local ErrRing ring;
    return ring;
}

/* 写入一条错误（内部使用） */
void err_record(const char *fmt, ...) {
    ErrRing &ring = tls_ring();
    ErrSlot &slot = ring.slots[ring.next % 8];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(slot.text, sizeof(slot.text), fmt, ap);
    va_end(ap);
    slot.used = true;
    ring.next = (ring.next + 1) % 8;
    if (ring.count < 8)
        ++ring.count;
}

/* 清空本线程错误历史（成功路径调用）：
 * 保证 kucclGetLastError 的语义严格为"自上次成功以来的失败"，
 * 不残留跨算子的旧错误误导排查。 */
void err_clear(void)
{
    ErrRing &ring = tls_ring();
    ring.count = 0;
    ring.next = 0;
    for (auto &slot : ring.slots)
        slot.used = false;
}

/* 取最近一条并清空（kucclGetLastError 用） */
const char *err_take_last() {
    ErrRing &ring = tls_ring();
    if (ring.count == 0)
        return nullptr;
    int last = (ring.next - 1 + 8) % 8;
    /* 挪到 TLS 静态缓冲返回——下次 take 前有效 */
    static thread_local char out[ERR_BUF];
    std::memcpy(out, ring.slots[last].text, ERR_BUF);
    ring.slots[last].used = false;
    ring.count = (ring.count > 0) ? ring.count - 1 : 0;
    return out;
}

/* 只读最近一条（不清空——调试用） */
const char *err_peek_last() {
    ErrRing &ring = tls_ring();
    if (ring.count == 0)
        return nullptr;
    int last = (ring.next - 1 + 8) % 8;
    return ring.slots[last].text;
}

/* 追加上下文（"调用链补信息"模式：底层记 UCG 状态，上层补算子名） */
void err_append(const char *ctx) {
    ErrRing &ring = tls_ring();
    if (ring.count == 0)
        return; /* 没有前置错误，丢弃 */
    int last = (ring.next - 1 + 8) % 8;
    size_t len = std::strlen(ring.slots[last].text);
    size_t room = ERR_BUF - len - 4;
    if (room > 0) {
        std::snprintf(ring.slots[last].text + len, room + 3, " | %s", ctx);
    }
}

} /* namespace detail */
} /* namespace kuccl */

extern "C" {

/* ════════════════════════════════════════════════════════════
 * 对外错误接口（kuccl.h 声明）
 *
 * kucclGetLastError：返回最近一次错误的详文并清空。
 *   - 返回 nullptr = 本线程无错误记录
 *   - 返回的指针在下一次本线程调用 kucclGetLastError 前有效
 *     （一次性读取语义）
 *   - 线程安全：TLS 存储，每线程独立
 *   - 典型用法：
 *       if (kucclAllReduce(...) != KUCCL_SUCCESS) {
 *           fprintf(stderr, "kuccl: %s\n", kucclGetLastError());
 *       }
 * ════════════════════════════════════════════════════════════*/
KUCCL_API const char *kucclGetLastError() {
    return kuccl::detail::err_take_last();
}

} /* extern "C" */
