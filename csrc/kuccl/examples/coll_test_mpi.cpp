/*
 * coll_test_mpi.cpp —— MPI 进程直接拉起 kuccl
 *
 * 覆盖接口（功能验证必测项）：
 *   kucclGetUniqueId / kucclCommInitRankConfig / kucclAllReduce /
 *   kucclBroadcast / kucclAllGather / kucclAllToAll / kucclTest /
 *   kucclCommDestroy；可选 --test-split 附加 kucclCommSplit 验证
 *
 *   1. config.oob_allgather 不能直接赋 MPI_Allgather（回调签名不同），
 *      需要包装函数把 kuccl 回调签名桥接到 MPI_Allgather
 *   2. kucclTest 的 status 出参是 kucclStatus_t*，不能传 MPI_STATUS_IGNORE
 *   3. 修正示例中 flag 重复声明 / request 变量名不一致 / h_recv 等笔误
 *
 * 编译运行命令见 RUN.md ⑤。
 */

#include <mpi.h>
#include <kuccl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define ELEM_COUNT 10

/* ── OOB 回调桥接：kuccl 回调签名 → MPI_Allgather ─────────────
 * 回调签名（types.hpp）：int(const void* send, void* recv, int count, void* ctx)
 * 语义对齐 MPI_Allgather：每 rank 贡献 count 字节，收齐后按 rank 拼接 */
static int mpi_oob_allgather(const void* sendbuf, void* recvbuf,
                             int count, void* ctx) {
    MPI_Comm comm = *static_cast<MPI_Comm*>(ctx);
    int rc = MPI_Allgather(sendbuf, count, MPI_BYTE,
                           recvbuf, count, MPI_BYTE, comm);
    return (rc == MPI_SUCCESS) ? 0 : -1;
}

/* ── kucclTest 轮询直至完成（请求推进发生在 kucclTest 内部）── */
static int wait_request(kucclRequest_t* req, int rank) {
    int flag = 0;
    kucclStatus_t st = KUCCL_STATUS_PENDING;
    while (!flag) {
        if (kucclTest(req, &flag, &st) != KUCCL_SUCCESS) {
            std::fprintf(stderr, "[Rank %d] kucclTest FAILED\n", rank);
            return -1;
        }
    }
    if (st != KUCCL_STATUS_COMPLETE) {
        std::fprintf(stderr, "[Rank %d] collective FAILED (status=%d)\n",
                     rank, static_cast<int>(st));
        return -1;
    }
    return 0;
}

static int g_failures = 0;

static void report(int rank, const char* op, int ok) {
    std::printf("[Rank %d] %-12s %s\n", rank, op, ok ? "PASS" : "FAIL");
    std::fflush(stdout);
    if (!ok) ++g_failures;
}

int main(int argc, char** argv) {
    const int want_split = (argc > 1 && std::strcmp(argv[1], "--test-split") == 0);

    /* ========== 1. MPI 初始化 ========== */
    MPI_Init(&argc, &argv);
    int rank = 0, world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    /* ========== 2. 生成并广播 kuccl 唯一 ID ========== */
    /* 所有进程必须使用同一个 ID 才能加入同一个通信域 */
    kucclUniqueId kuccl_id;
    std::memset(&kuccl_id, 0, sizeof(kuccl_id));
    if (rank == 0) {
        if (kucclGetUniqueId(&kuccl_id) != KUCCL_SUCCESS) {
            std::fprintf(stderr, "[Rank 0] kucclGetUniqueId FAILED\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(&kuccl_id, sizeof(kuccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    /* ========== 3. 创建 kuccl 通信域（集合操作，全员同时调用）========== */
    MPI_Comm oob_comm = MPI_COMM_WORLD;   /* oob_ctx 指向它，回调里解引用 */
    kucclConfig_t config;
    std::memset(&config, 0, sizeof(config));
    config.oob_allgather = mpi_oob_allgather;
    config.oob_bcast     = nullptr;       /* 选填：库内广播走 allgather 语义 */
    config.oob_ctx       = &oob_comm;

    kucclComm_t comm = nullptr;
    if (kucclCommInitRankConfig(&comm, world, kuccl_id, rank, &config)
            != KUCCL_SUCCESS) {
        std::fprintf(stderr, "[Rank %d] kucclCommInitRankConfig FAILED\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::printf("[Rank %d] 通信域已创建 (nranks=%d)\n", rank, world);
    std::fflush(stdout);

    /* ========== 4. AllReduce：send[i]=rank，期望 sum=world*(world-1)/2 ========== */
    {
        float* send = static_cast<float*>(std::malloc(ELEM_COUNT * sizeof(float)));
        float* recv = static_cast<float*>(std::malloc(ELEM_COUNT * sizeof(float)));
        for (int i = 0; i < ELEM_COUNT; ++i) send[i] = static_cast<float>(rank);

        kucclRequest_t req = nullptr;
        int ok = 0;
        if (kucclAllReduce(send, recv, ELEM_COUNT, kucclFloat, kucclSum,
                           comm, &req) == KUCCL_SUCCESS &&
            wait_request(&req, rank) == 0) {
            const float expected = static_cast<float>(world * (world - 1) / 2);
            ok = 1;
            for (int i = 0; i < ELEM_COUNT; ++i)
                if (recv[i] != expected) { ok = 0; break; }
            std::printf("[Rank %d] AllReduce 首元素=%.1f 预期=%.1f\n",
                        rank, recv[0], expected);
        }
        report(rank, "AllReduce", ok);
        std::free(send);
        std::free(recv);
    }

    /* ========== 5. Broadcast：root 填 42，期望全员 42 ==========
     * 库语义：sendbuff 仅 root 有效（root 内部先拷到 recvbuff），
     * 非 root 传 recvbuff 占位即可 */
    {
        const int root = 0;
        int32_t* recv = static_cast<int32_t*>(
            std::malloc(ELEM_COUNT * sizeof(int32_t)));
        int32_t* send = static_cast<int32_t*>(
            std::malloc(ELEM_COUNT * sizeof(int32_t)));
        for (int i = 0; i < ELEM_COUNT; ++i) send[i] = 42;

        kucclRequest_t req = nullptr;
        int ok = 0;
        if (kucclBroadcast(send, recv, ELEM_COUNT, kucclInt32, root,
                           comm, &req) == KUCCL_SUCCESS &&
            wait_request(&req, rank) == 0) {
            ok = 1;
            for (int i = 0; i < ELEM_COUNT; ++i)
                if (recv[i] != 42) { ok = 0; break; }
        }
        report(rank, "Broadcast", ok);
        std::free(send);
        std::free(recv);
    }

    /* ========== 6. AllGather：rank r 贡献 [r,r,...]，期望 recv=[0..ws-1] ========== */
    {
        int32_t* send = static_cast<int32_t*>(
            std::malloc(ELEM_COUNT * sizeof(int32_t)));
        int32_t* recv = static_cast<int32_t*>(
            std::malloc(static_cast<size_t>(world) * ELEM_COUNT * sizeof(int32_t)));
        for (int i = 0; i < ELEM_COUNT; ++i) send[i] = rank;

        kucclRequest_t req = nullptr;
        int ok = 0;
        if (kucclAllGather(send, recv, ELEM_COUNT, kucclInt32,
                           comm, &req) == KUCCL_SUCCESS &&
            wait_request(&req, rank) == 0) {
            ok = 1;
            for (int j = 0; j < world && ok; ++j)
                for (int i = 0; i < ELEM_COUNT; ++i)
                    if (recv[j * ELEM_COUNT + i] != j) { ok = 0; break; }
        }
        report(rank, "AllGather", ok);
        std::free(send);
        std::free(recv);
    }

    /* ========== 7. AllToAll：r 发往 j 的块填 r*100+j，期望 recv 第 j 块 = j*100+r ========== */
    {
        const int per_peer = 2;
        const int total = world * per_peer;
        int32_t* send = static_cast<int32_t*>(std::malloc(total * sizeof(int32_t)));
        int32_t* recv = static_cast<int32_t*>(std::malloc(total * sizeof(int32_t)));
        for (int j = 0; j < world; ++j)
            for (int i = 0; i < per_peer; ++i)
                send[j * per_peer + i] = rank * 100 + j;

        kucclRequest_t req = nullptr;
        int ok = 0;
        if (kucclAllToAll(send, recv, per_peer, kucclInt32,
                          comm, &req) == KUCCL_SUCCESS &&
            wait_request(&req, rank) == 0) {
            ok = 1;
            for (int j = 0; j < world && ok; ++j)
                for (int i = 0; i < per_peer; ++i)
                    if (recv[j * per_peer + i] != j * 100 + rank) { ok = 0; break; }
        }
        report(rank, "AllToAll", ok);
        std::free(send);
        std::free(recv);
    }

    /* ========== 8.（可选）CommSplit：color=rank%2，子组 AllReduce ==========
     * 子组成员按 global rank 排序 → 本进程新 rank = rank/2，
     * 子组大小 m = (rank 为偶) ? (world+1)/2 : world/2，
     * 期望子组 sum = m*(m-1)/2。
     * 注意：V2 的 CommSplit 不支持 color<0（不加入），所有进程必须给有效 color */
    if (want_split) {
        kucclComm_t sub = nullptr;
        const int color = rank % 2;
        int ok = 0;
        if (kucclCommSplit(comm, color, rank, &sub, nullptr) == KUCCL_SUCCESS) {
            const int new_rank = rank / 2;
            const int m = (color == 0) ? (world + 1) / 2 : world / 2;
            float* send = static_cast<float*>(std::malloc(sizeof(float)));
            float* recv = static_cast<float*>(std::malloc(sizeof(float)));
            send[0] = static_cast<float>(new_rank);

            kucclRequest_t req = nullptr;
            if (kucclAllReduce(send, recv, 1, kucclFloat, kucclSum,
                               sub, &req) == KUCCL_SUCCESS &&
                wait_request(&req, rank) == 0) {
                const float expected =
                    static_cast<float>(m * (m - 1) / 2);
                ok = (recv[0] == expected);
                std::printf("[Rank %d] Split(color=%d) sub-allreduce=%.1f 预期=%.1f\n",
                            rank, color, recv[0], expected);
            }
            std::free(send);
            std::free(recv);
            kucclCommDestroy(sub);
        }
        report(rank, "CommSplit", ok);
    }

    /* ========== 9. 资源释放 ========== */
    kucclCommDestroy(comm);
    std::printf("[Rank %d] 通信域已销毁\n", rank);
    std::fflush(stdout);

    const int final_fail = g_failures;
    MPI_Reduce(&final_fail, &g_failures, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        std::printf("=== MPI 直调用例 %s（失败项=%d）===\n",
                    g_failures == 0 ? "ALL PASSED" : "SOME FAILED", g_failures);

    MPI_Finalize();
    return (g_failures == 0) ? 0 : 1;
}
