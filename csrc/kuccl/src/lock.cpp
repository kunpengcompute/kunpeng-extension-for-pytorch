/*
 * lock.cpp —— 公开接口的 API 串行化锁（声明见 comm_impl.hpp）
 *
 * 后端底层资源以 SINGLE 线程模式创建，公开接口在入口用这把锁
 * 做外部串行化（SINGLE 模式的标准用法）。
 * recursive_mutex：kucclCommSplit 内部重入 kucclCommInitRankConfig。
 */

#include "comm_impl.hpp"

namespace kuccl {
namespace detail {

std::recursive_mutex g_api_lock;

} /* namespace detail */
} /* namespace kuccl */
