/*
 * bench_a2a_mpi.cpp —— MPI 拉起的 kuccl alltoall 微基准（native vs direct）
 * alltoall 无共享无归约，直拉即跨界流量下界；两种算法的差别在
 * 同步/拷回/黑盒开销。
 * 数据：send[src][dst][m] = (src * W + dst) * count + m（逐元素 VERIFY）。
 * argv: <min_shift> <max_shift> <iters> [warmup]
 *   尺寸扫描 2^min ~ 2^max 字节（per-rank chunk 大小）。
 */

#include <mpi.h>
#include <kuccl.h>

#include <chrono>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int mpi_oob_allgather(const void *sendbuf, void *recvbuf, int count, void *ctx)
{
    MPI_Comm comm = *static_cast<MPI_Comm *>(ctx);
    return MPI_Allgather(sendbuf, count, MPI_BYTE, recvbuf, count, MPI_BYTE, comm) == MPI_SUCCESS ? 0 : -1;
}

static double now_us()
{
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void request_failed(int rank, const char *where)
{
    const char *error = kucclGetLastError();
    std::fprintf(stderr, "[Rank %d] %s FAILED: %s\n", rank, where, error ? error : "request did not complete");
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

int main(int argc, char **argv)
{
    const int min_shift = argc > 1 ? std::atoi(argv[1]) : 3;
    const int max_shift = argc > 2 ? std::atoi(argv[2]) : 20;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 30;
    const int warmup = argc > 4 ? std::atoi(argv[4]) : 5;

    MPI_Init(&argc, &argv);
    int rank = 0, world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    if (min_shift < 2 || max_shift < min_shift || max_shift >= static_cast<int>(sizeof(size_t) * CHAR_BIT) ||
        iters <= 0 || warmup < 0) {
        if (rank == 0)
            std::fprintf(stderr, "Invalid sizes/iterations: require 2 <= min_shift <= max_shift < size_t bits, iters > "
                                 "0, warmup >= 0\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    kucclUniqueId kuccl_id;
    std::memset(&kuccl_id, 0, sizeof(kuccl_id));
    if (rank == 0 && kucclGetUniqueId(&kuccl_id) != KuCCL_SUCCESS)
        MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Bcast(&kuccl_id, sizeof(kuccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    MPI_Comm oob_comm = MPI_COMM_WORLD;
    kucclConfig_t config;
    std::memset(&config, 0, sizeof(config));
    config.oob_allgather = mpi_oob_allgather;
    config.oob_ctx = &oob_comm;

    kucclComm_t comm = nullptr;
    if (kucclCommInitRankConfig(&comm, world, kuccl_id, rank, &config) != KuCCL_SUCCESS) {
        const char *init_err = kucclGetLastError(); /* one-shot API: take once, print the copy */
        std::fprintf(stderr, "[Rank %d] kucclCommInitRankConfig FAILED: %s\n", rank,
                     init_err ? init_err : "no error detail");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        std::printf("%10s %12s %12s %12s\n", "size", "avg_us", "algbw_G", "busbw_G");
        std::fflush(stdout);
    }

    int failures = 0;
    for (int shift = min_shift; shift <= max_shift; shift++) {
        const size_t bytes = static_cast<size_t>(1) << shift; /* per-rank chunk bytes */
        const size_t count = bytes / sizeof(int32_t);
        /* Bound the full buffer and the source/destination/element encoding
         * before multiplying or allocating; the benchmark uses int32 values. */
        if (count > static_cast<size_t>(INT32_MAX) / sizeof(int32_t) / world ||
            count > static_cast<size_t>(INT32_MAX) / world / world) {
            if (rank == 0)
                std::fprintf(stderr, "AllToAll size/encoding exceeds int32 range at %zuB\n", bytes);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        std::vector<int32_t> send(count * world, 0);
        std::vector<int32_t> recv(count * world, -1);
        /* s is the destination block; rank identifies this sender. */
        for (size_t s = 0; s < (size_t)world; s++)
            for (size_t m = 0; m < count; m++)
                send[s * count + m] = static_cast<int32_t>((static_cast<size_t>(rank) * world + s) * count + m);

        auto exchange = [&] {
            kucclRequest_t req = nullptr;
            if (kucclAllToAll(send.data(), recv.data(), count, kucclInt32, comm, &req) != KuCCL_SUCCESS) {
                request_failed(rank, "AllToAll");
            }
            int flag = 0;
            kucclStatus_t st = KuCCL_STATUS_PENDING;
            while (!flag) {
                if (kucclTest(&req, &flag, &st) != KuCCL_SUCCESS)
                    request_failed(rank, "Test");
            }
            if (st != KuCCL_STATUS_COMPLETE)
                request_failed(rank, "Test status");
        };
        auto verify = [&] {
            int local_bad = 0;
            for (size_t i = 0; i < recv.size(); i++) {
                const int32_t expect = static_cast<int32_t>(((i / count) * world + rank) * count + (i % count));
                if (recv[i] != expect) {
                    std::fprintf(stderr, "[Rank %d] VERIFY FAIL @%zuB: recv[%zu]=%d expect=%d\n", rank, bytes, i,
                                 recv[i], expect);
                    local_bad = 1;
                    break;
                }
            }
            int global_bad = 0;
            MPI_Allreduce(&local_bad, &global_bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            return global_bad == 0;
        };

        /* Even warmup=0 must execute one unmeasured correctness exchange. */
        for (int i = 0; i < std::max(1, warmup); i++)
            exchange();
        if (!verify()) {
            failures = 1;
            break;
        }

        std::fill(recv.begin(), recv.end(), -1);
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = now_us();
        for (int i = 0; i < iters; i++)
            exchange();
        const double local_avg = (now_us() - t0) / iters;
        if (!verify()) {
            failures = 1;
            break;
        }
        double avg = 0;
        MPI_Allreduce(&local_avg, &avg, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (rank == 0) {
            /* 每 rank 收发 (W−1) 份 chunk；alltoall 无归约因子，busbw = algbw */
            const double algbw = static_cast<double>(bytes) * (world - 1) / (avg * 1e-6) / 1e9;
            std::printf("%10zu %12.2f %12.3f %12.3f\n", bytes, avg, algbw, algbw);
            std::fflush(stdout);
        }
    }

    if (rank == 0)
        std::printf("%s\n", failures == 0 ? "VERIFY: ALL PASSED" : "VERIFY: FAILED");
    kucclCommDestroy(comm);
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
