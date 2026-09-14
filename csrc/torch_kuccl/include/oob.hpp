#ifndef KUCCL_OOB_HPP_
#define KUCCL_OOB_HPP_

/*
 * OOB（Out-of-Band）引导层
 *
 * 在 UCX 高速通道建立之前，基于 c10d::TCPStore 交换各进程的 UCX worker 地址。
 * 这是"低速小路"，建组后不再使用（除非显式 barrier）。
 */

#include <torch/csrc/distributed/c10d/Store.hpp>

#include <cstddef>
#include <string>

namespace kuccl {

/* OOB AllGather（语义对齐 MPI_Allgather）：
 *   每个 rank 把 sendbuf[0..msglen) 发布到 Store，
 *   rank-0 聚合后广播 packed result，其余 rank 等待获取。
 *   recvbuf = world_size * msglen 字节，slot[i] = rank-i 数据。
 *   prefix 用于多子组隔离（由调用方传入唯一前缀）。 */
void oob_allgather(c10d::Store *store, const void *sendbuf, void *recvbuf, int rank, int world_size,
                   std::size_t msglen, const std::string &prefix = "");

/* 基于 Store key 的全局屏障。
 * 在销毁 TCPStore server 前调用，防止 server 先析构导致 peer 连接断开。 */
void store_barrier(c10d::Store *store, int rank, int world_size, const std::string &tag);

} // namespace kuccl

#endif /* KUCCL_OOB_HPP_ */
