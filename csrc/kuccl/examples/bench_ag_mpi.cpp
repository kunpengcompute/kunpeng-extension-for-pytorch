/*
 * bench_ag_mpi.cpp —— MPI 拉起的 kuccl allgather 微基准
 * mpirun --map-by/--bind-to 组合拓扑布局，对比 linear / hier_die /
 * hier_socket 在真实聚簇布局下的表现（矩阵见 bench_ag_mpi.sh）。
 * argv: <min_shift> <max_shift> <iters> [warmup]
 *   尺寸扫描 2^min ~ 2^max 字节（uint8），每档 warmup 热身 + iters 计时。
 */

#include <mpi.h>
#include <kuccl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* OOB 桥接：kuccl 回调签名 → MPI_Allgather（同 coll_test_mpi） */
static int mpi_oob_allgather(const void *sendbuf, void *recvbuf, int count, void *ctx)
{
    MPI_Comm comm = *static_cast<MPI_Comm *>(ctx);
    return MPI_Allgather(sendbuf, count, MPI_BYTE, recvbuf, count, MPI_BYTE, comm) == MPI_SUCCESS ? 0 : -1;
}

static double now_us()
{
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
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

    /* kuccl 通信域（OOB = MPI_Allgather 桥接） */
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
        std::fprintf(stderr, "[Rank %d] kucclCommInitRankConfig FAILED\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        std::printf("%10s %12s %12s %12s\n", "size", "avg_us", "algbw_G", "busbw_G");
        std::fflush(stdout);
    }

    int failures = 0;
    for (int shift = min_shift; shift <= max_shift; shift++) {
        const size_t bytes = static_cast<size_t>(1) << shift;
        std::vector<uint8_t> send(bytes, static_cast<uint8_t>(rank));
        std::vector<uint8_t> recv(static_cast<size_t>(world) * bytes, 0xFF);

        kucclRequest_t req = nullptr;
        int flag = 0;
        kucclStatus_t st = KuCCL_STATUS_PENDING;

        /* 热身 + 首次正确性校验 */
        for (int i = 0; i < warmup; i++) {
            if (kucclAllGather(send.data(), recv.data(), bytes, kucclUint8, comm, &req) != KuCCL_SUCCESS) {
                std::fprintf(stderr, "[Rank %d] AllGather FAILED at %zuB\n", rank, bytes);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            flag = 0;
            while (!flag)
                kucclTest(&req, &flag, &st);
        }
        for (size_t i = 0; i < recv.size(); i++) {
            const uint8_t expect = static_cast<uint8_t>(i / bytes);
            if (recv[i] != expect) {
                if (rank == 0)
                    std::fprintf(stderr, "VERIFY FAIL @%zuB: recv[%zu]=%u expect=%u\n", bytes, i, recv[i], expect);
                ++failures;
                break;
            }
        }

        /* 计时：MPI barrier 对齐后连续 iters 次 */
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = now_us();
        for (int i = 0; i < iters; i++) {
            kucclAllGather(send.data(), recv.data(), bytes, kucclUint8, comm, &req);
            flag = 0;
            while (!flag)
                kucclTest(&req, &flag, &st);
        }
        const double avg = (now_us() - t0) / iters;

        if (rank == 0) {
            const double total_bytes = static_cast<double>(bytes) * world;
            const double algbw = total_bytes / (avg * 1e-6) / 1e9;
            const double busbw = algbw * (world - 1.0) / world;
            std::printf("%10zu %12.2f %12.3f %12.3f\n", bytes, avg, algbw, busbw);
            std::fflush(stdout);
        }
    }

    if (rank == 0)
        std::printf("%s\n", failures == 0 ? "VERIFY: ALL PASSED" : "VERIFY: FAILED");
    kucclCommDestroy(comm);
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
