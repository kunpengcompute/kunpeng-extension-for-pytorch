/*
 * KuCCL OOB 实现 —— 基于 c10d::TCPStore 的引导通信
 *
 * 逻辑：
 *   - leader-then-broadcast allgather（rank-0 聚合后广播）
 *   - store-based barrier
 */

#include "oob.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <mutex>
#include <string>
#include <vector>

#include <unistd.h>

namespace kuccl {

namespace {

std::mutex g_log_mutex;

bool oob_debug_enabled()
{
    static const bool enabled = [] {
        const char *v = std::getenv("TORCH_KUCCL_OOB_DEBUG");
        return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    }();
    return enabled;
}

std::string get_hostname()
{
    char buf[256];
    std::memset(buf, 0, sizeof(buf));
    if (::gethostname(buf, sizeof(buf) - 1) != 0)
        return "unknown";
    return std::string(buf);
}

/* 输出走 stderr：stdout 是 benchmark 的解析流，调试输出绝不污染它 */
void log_oob(int rank, int ws, const std::string &prefix, const std::string &msg)
{
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
                  .count();
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::cerr << "[KuCCL-OOB][t=" << ms << "ms][host=" << get_hostname() << "][pid=" << static_cast<int>(::getpid())
              << "][rank=" << rank << "/" << ws << "][prefix=" << prefix << "] " << msg << std::endl;
}

} // anonymous namespace

void oob_allgather(c10d::Store *store, const void *sendbuf, void *recvbuf, int rank, int world_size, std::size_t msglen,
                   const std::string &prefix)
{
    if (!store)
        throw std::invalid_argument("oob_allgather: store is null");
    if (!sendbuf || !recvbuf)
        throw std::invalid_argument("oob_allgather: null buffer");
    if (world_size <= 0 || rank < 0 || rank >= world_size)
        throw std::invalid_argument("oob_allgather: invalid rank/world_size");
    if (msglen == 0)
        throw std::invalid_argument("oob_allgather: msglen == 0");

    const bool debug = oob_debug_enabled();
    constexpr int kTimeoutMs = 90'000;
    const std::chrono::milliseconds timeout(kTimeoutMs);
    const std::size_t result_size = static_cast<std::size_t>(world_size) * msglen;

    const std::string input_prefix = prefix + "input_";
    const std::string my_key = input_prefix + std::to_string(rank);
    const std::string result_key = prefix + "result";

    if (debug)
        log_oob(rank, world_size, prefix, "ENTER msglen=" + std::to_string(msglen));

    /* Step 1: 所有 rank 发布自己的数据 */
    const auto *src = static_cast<const uint8_t *>(sendbuf);
    store->set(my_key, std::vector<uint8_t>(src, src + msglen));

    /* Step 2: rank-0 聚合后广播 */
    if (rank == 0) {
        std::vector<std::string> input_keys;
        input_keys.reserve(static_cast<std::size_t>(world_size));
        for (int i = 0; i < world_size; ++i)
            input_keys.push_back(input_prefix + std::to_string(i));

        store->wait(input_keys, timeout);

        std::vector<uint8_t> packed(result_size);
        for (int i = 0; i < world_size; ++i) {
            auto value = store->get(input_keys[i]);
            if (value.size() != msglen)
                throw std::runtime_error("oob_allgather: size mismatch peer=" + std::to_string(i));
            std::memcpy(packed.data() + static_cast<std::size_t>(i) * msglen, value.data(), msglen);
        }
        store->set(result_key, packed);
        std::memcpy(recvbuf, packed.data(), result_size);

        if (debug) {
            log_oob(rank, world_size, prefix, "COMPLETE_LEADER result_bytes=" + std::to_string(result_size));
        }
        return;
    }

    /* Step 3: 其他 rank 等待 result */
    store->wait(std::vector<std::string>{result_key}, timeout);
    auto result = store->get(result_key);
    if (result.size() != result_size)
        throw std::runtime_error("oob_allgather: packed result size mismatch");
    std::memcpy(recvbuf, result.data(), result_size);

    if (debug) {
        log_oob(rank, world_size, prefix, "COMPLETE_FOLLOWER result_bytes=" + std::to_string(result_size));
    }
}

void store_barrier(c10d::Store *store, int rank, int world_size, const std::string &tag)
{
    if (!store)
        throw std::invalid_argument("store_barrier: store is null");
    if (world_size <= 0 || rank < 0 || rank >= world_size)
        throw std::invalid_argument("store_barrier: invalid rank/world_size");
    if (world_size == 1)
        return;

    const std::string prefix = "barrier_" + tag;
    const std::string my_key = prefix + "_" + std::to_string(rank);

    store->set(my_key, std::vector<uint8_t>{1});

    std::vector<std::string> all_keys;
    all_keys.reserve(static_cast<std::size_t>(world_size));
    for (int i = 0; i < world_size; ++i)
        all_keys.push_back(prefix + "_" + std::to_string(i));

    store->wait(all_keys);
}

} // namespace kuccl
