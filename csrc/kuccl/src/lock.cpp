/*
 * lock.cpp —— API 串行化锁的定义（声明见 comm_impl.hpp）
 *
 * 串行化锁：后端以 SINGLE 线程模式创建底层资源（xucg：UCP worker
 * SINGLE 模式 + UCG SINGLE 初始化），库层不为线程安全付出运行时代价，
 * 要求调用方保证"同一时刻只有一个线程在库内"。
 * 适配层（torch_kuccl）经 Python 调用时天然被 GIL 串行化；
 * 但 C++ 调用方（MPI 直调 / 多线程 c10d）没有这层保护。
 * 因此全部公开接口（算子 + 通信域管理 + kucclTest）在入口用这把锁
 * 做外部串行化——这是 SINGLE 模式的标准用法（serialized access）。
 *
 * 用 recursive_mutex 的原因：kucclCommSplit 内部会调
 * kucclCommInitRankConfig（重入）。
 */

#include "comm_impl.hpp"

namespace kuccl {
namespace detail {

std::recursive_mutex g_api_lock;

} /* namespace detail */
} /* namespace kuccl */
