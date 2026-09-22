/*
 * coll_test.cpp V2 — C++ 端 AllGather 集合通信测试
 *
 * 重构说明（vs V1）：
 *   - V1 用 kuccl::team_create（OOB 回调直接注入）→ V2 用设计书接口
 *     kucclGetUniqueId + kucclCommInitRankConfig（config 注入 OOB 回调）
 *   - V1 阻塞式 kuccl_allgather → V2 异步 kucclAllGather + kucclTest 轮询
 *   - 本测试演示"调用方提供 OOB"模式：用 TCPStore 实现 OOB allgather，
 *     封装成 kucclOobAllgatherFn 回调注入（对齐设计书 MPI 直调用例）
 *
 * 运行方式：
 *   rank0: ./coll_test 0 2 127.0.0.1
 *   rank1: ./coll_test 1 2 127.0.0.1
 */

#include <kuccl.h>                 /* libkuccl 对外接口 */

#include <torch/csrc/distributed/c10d/TCPStore.hpp>   /* 仅测试用（OOB 通道） */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

/* ════════════════════════════════════════════════════════════
 * 测试用 OOB 上下文与回调（"调用方提供 OOB"模式）
 * ════════════════════════════════════════════════════════════*/

struct TestOobCtx {
    c10d::Store* store;
    int          rank;
    int          world_size;
    uint64_t     round;      /* 轮次计数（Store key 隔离） */
};

/* OOB allgather 回调：leader-then-broadcast on TCPStore
 * 签名对齐 kucclOobAllgatherFn：int(const void* send, void* recv, int count, void* ctx) */
static int test_oob_allgather(const void* sendbuf, void* recvbuf,
                               int count, void* user_ctx) {
    auto* ctx = static_cast<TestOobCtx*>(user_ctx);
    if (!ctx || !ctx->store) return -1;

    const uint64_t r = ctx->round++;
    const std::string prefix = "test_oob_" + std::to_string(r) + "_";
    const auto* src = static_cast<const uint8_t*>(sendbuf);
    auto* dst = static_cast<uint8_t*>(recvbuf);

    /* 所有 rank 发布自己的数据 */
    ctx->store->set(prefix + "in_" + std::to_string(ctx->rank),
                    std::vector<uint8_t>(src, src + count));

    if (ctx->rank == 0) {
        /* rank 0 聚合后广播 */
        std::vector<std::string> keys;
        for (int i = 0; i < ctx->world_size; ++i)
            keys.push_back(prefix + "in_" + std::to_string(i));
        ctx->store->wait(keys, std::chrono::milliseconds(90000));
        for (int i = 0; i < ctx->world_size; ++i) {
            auto v = ctx->store->get(keys[i]);
            std::memcpy(dst + static_cast<size_t>(i) * count, v.data(), count);
        }
        ctx->store->set(prefix + "result",
                        std::vector<uint8_t>(dst, dst + static_cast<size_t>(ctx->world_size) * count));
    } else {
        /* 其他 rank 等结果 */
        ctx->store->wait({prefix + "result"}, std::chrono::milliseconds(90000));
        auto v = ctx->store->get(prefix + "result");
        std::memcpy(dst, v.data(), v.size());
    }
    return 0;
}

/* ════════════════════════════════════════════════════════════
 * 主流程
 * ════════════════════════════════════════════════════════════*/

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::fprintf(stderr,
            "用法: %s <rank> <world_size> <server_ip>\n"
            "  rank0: %s 0 2 127.0.0.1\n"
            "  rank1: %s 1 2 127.0.0.1\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    const int          rank       = std::atoi(argv[1]);
    const int          world_size = std::atoi(argv[2]);
    const std::string  server_ip  = argv[3];
    constexpr uint16_t STORE_PORT = 23457;

    /* ── 1. TCPStore（rank-0 为 server，仅作 OOB 通道）── */
    std::shared_ptr<c10d::Store> store;
    try {
        c10d::TCPStoreOptions opts;
        opts.port        = STORE_PORT;
        opts.isServer    = (rank == 0);
        opts.numWorkers  = world_size;
        opts.waitWorkers = true;
        opts.timeout     = std::chrono::seconds(60);
        store = std::make_shared<c10d::TCPStore>(server_ip, opts);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[Rank %d] TCPStore 初始化失败: %s\n", rank, e.what());
        return 1;
    }
    std::printf("[Rank %d] TCPStore 已连接\n", rank);

    /* ── 2. 分发 uniqueId（rank 0 生成，Store 广播）── */
    kucclUniqueId uniqueId;
    std::memset(&uniqueId, 0, sizeof(uniqueId));
    if (rank == 0) {
        if (kucclGetUniqueId(&uniqueId) != KUCCL_SUCCESS) {
            std::fprintf(stderr, "[Rank %d] kucclGetUniqueId 失败\n", rank);
            return 1;
        }
        store->set("test_uniqueid",
                   std::vector<uint8_t>(reinterpret_cast<uint8_t*>(&uniqueId),
                                        reinterpret_cast<uint8_t*>(&uniqueId) + sizeof(uniqueId)));
    } else {
        store->wait({"test_uniqueid"}, std::chrono::milliseconds(60000));
        auto v = store->get("test_uniqueid");
        std::memcpy(uniqueId.data, v.data(), sizeof(uniqueId));
    }
    std::printf("[Rank %d] uniqueId 已分发\n", rank);

    /* ── 3. 构造 config（注入 OOB 回调）+ 创建通信域 ── */
    TestOobCtx oob_ctx;
    oob_ctx.store      = store.get();
    oob_ctx.rank       = rank;
    oob_ctx.world_size = world_size;
    oob_ctx.round      = 0;

    kucclConfig_t config;
    std::memset(&config, 0, sizeof(config));
    config.oob_allgather = &test_oob_allgather;
    config.oob_ctx       = &oob_ctx;

    kucclComm_t comm = nullptr;
    if (kucclCommInitRankConfig(&comm, world_size, uniqueId, rank, &config)
            != KUCCL_SUCCESS) {
        std::fprintf(stderr, "[Rank %d] kucclCommInitRankConfig 失败\n", rank);
        return 1;
    }
    std::printf("[Rank %d] 通信域已创建\n", rank);

    /* ── 4. AllGather（异步：start + Test 轮询）── */
    const size_t count = 4;
    std::vector<int> sendbuf(count);
    std::vector<int> recvbuf(count * static_cast<size_t>(world_size), -1);
    for (size_t i = 0; i < count; ++i)
        sendbuf[i] = rank * 10 + static_cast<int>(i);

    kucclRequest_t req = nullptr;
    if (kucclAllGather(sendbuf.data(), recvbuf.data(),
                        count, kucclInt32, comm, &req) != KUCCL_SUCCESS) {
        std::fprintf(stderr, "[Rank %d] kucclAllGather 启动失败\n", rank);
        kucclCommDestroy(comm);
        return 1;
    }

    /* kucclTest 轮询直到完成（内部推进 worker） */
    int flag = 0;
    kucclStatus_t status = KUCCL_STATUS_PENDING;
    while (!flag) {
        if (kucclTest(&req, &flag, &status) != KUCCL_SUCCESS) {
            std::fprintf(stderr, "[Rank %d] kucclTest 失败\n", rank);
            kucclCommDestroy(comm);
            return 1;
        }
    }
    if (status != KUCCL_STATUS_COMPLETE) {
        std::fprintf(stderr, "[Rank %d] AllGather 出错 (status=%d)\n", rank, (int)status);
        kucclCommDestroy(comm);
        return 1;
    }

    /* ── 5. 打印 + 校验 ── */
    std::printf("[Rank %d] recvbuf = [", rank);
    for (size_t i = 0; i < recvbuf.size(); ++i)
        std::printf("%d%s", recvbuf[i], (i + 1 < recvbuf.size()) ? ", " : "");
    std::printf("]\n");

    bool ok = true;
    for (int r = 0; r < world_size; ++r) {
        for (size_t i = 0; i < count; ++i) {
            const int got = recvbuf[static_cast<size_t>(r) * count + i];
            const int exp = r * 10 + static_cast<int>(i);
            if (got != exp) {
                std::fprintf(stderr,
                    "[Rank %d] 数据不一致: rank=%d idx=%zu got=%d expected=%d\n",
                    rank, r, i, got, exp);
                ok = false;
            }
        }
    }
    std::printf("[Rank %d] allgather %s\n", rank, ok ? "PASSED" : "FAILED");

    /* ── 6. 销毁（全员 barrier 后再销毁，保护 Store server）── */
    {
        const std::string my_key = "coll_done_" + std::to_string(rank);
        store->set(my_key, std::vector<uint8_t>{1});
        std::vector<std::string> all_keys;
        for (int i = 0; i < world_size; ++i)
            all_keys.push_back("coll_done_" + std::to_string(i));
        store->wait(all_keys, std::chrono::milliseconds(60000));
    }
    kucclCommDestroy(comm);
    std::printf("[Rank %d] 通信域已销毁\n", rank);
    return ok ? 0 : 2;
}
