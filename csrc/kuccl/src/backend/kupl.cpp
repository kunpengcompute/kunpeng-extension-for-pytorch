/*
 * kupl backend — single-node shared-memory fast path over KUPL shm.
 *
 * User buffers are plain memory; KUPL requires them inside kupl_shm_win.
 * Each comm keeps lazily-grown staging windows; all copies go through
 * kupl_memcpy (SDMA if available, CPU otherwise).
 *
 * AllGather: KUCCL_KUPL_AG_ALGO linear | auto | hier | hier_die | hier_socket | smart,
 *   sync per KUCCL_KUPL_AG_SYNC fence | doorbell. Default (unset) is linear
 *   direct reads (allgather lower bound); auto/hier route by topology —
 *   layered pair where a boundary exists (socket first, die second), linear
 *   otherwise; hier_die/hier_socket pin the granularity; smart learns per
 *   size bucket on repeated calls.
 *
 * AllReduce: KUCCL_KUPL_AR_ALGO auto | flat | hier. flat (default, same as
 *   unset) keeps SUM on KUPL native (recursive halving/doubling) and MIN/MAX
 *   on RH-RD or direct reads; hier runs the layered
 *   reduce-scatter/o/e-exchange/fanout at any size; auto is size-routed
 *   (hier past KUCCL_KUPL_AR_HIER_MIN, flat below).
 *
 * AllToAll: KUCCL_KUPL_A2A_ALGO direct | native. direct (default) stages
 * once and pulls every chunk straight into the user recvbuf — no native
 * call, no copy-back, 2 fences; native rides KUPL's alltoall for A/B
 * against the installed KUPL version.
 *
 * Safety: all ranks issue the same op sequence with the same (count,
 * datatype). Window growth derives only from collective parameters.
 * Flag rounds drain before overwriting. Failures advance the round.
 */

#include "backend.hpp" /* err_record / g_api_lock (via comm_impl.hpp) */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

#if defined(KuCCL_HAVE_KUPL)

#include <kupl.h>

#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>

namespace kuccl {
namespace backend {

namespace {

enum AllgatherAlgo {
    ALLGATHER_LINEAR = 0,      /* flat direct reads; sync per KUCCL_KUPL_AG_SYNC */
    ALLGATHER_DIE_PAIR = 1,    /* pair exchange at the die boundary (fence-structured) */
    ALLGATHER_SOCKET_PAIR = 2, /* pair exchange at the socket boundary (fence-structured) */
};

/* A routable (structure, sync) pair — smart mode's unit of choice. */
struct AgChoice {
    uint8_t algo = 0;      /* AllgatherAlgo */
    bool doorbell = false; /* linear path only */
};

enum AllreduceAlgo {
    ALLREDUCE_FLAT = 0,        /* SUM→KUPL native RH-RD; MIN/MAX→RH-RD or direct */
    ALLREDUCE_HIER_DIE = 1,    /* hier @ die (NUMA node) granularity — internal pick */
    ALLREDUCE_HIER_SOCKET = 2, /* hier @ socket (package) granularity — internal pick */
};

struct Config {
    int forced_algo = -1;                    /* -1=unforced; 0=linear 1=hier_die 2=hier_socket */
    bool topo_auto = false;                  /* explicit "auto"/"hier": topology-routed (unset default is linear) */
    bool sync_doorbell = false;              /* KUCCL_KUPL_AG_SYNC: linear path's sync primitive */
    bool smart_route = false;                /* explicit "smart": lazy per-bucket learning */
    int smart_reps = 2;                      /* smart mode: samples per candidate during exploration */
    int parallel_threads = 4;                /* executor threads for reads/fanout */
    uint64_t parallel_min_bytes = 16 * 1024; /* linear read parallel threshold */
    bool peer_map_own = true;                /* build peer mappings ourselves (see cache_peer_bases) */
    int pf_dist = 512;                       /* prefetch distance (bytes) for combine
                                                loops; 0 disables */
    uint64_t spin_max = 200000;              /* spin budget before sched_yield */
    uint64_t scratch_floor = 2 << 20;        /* first staging allocation floor */
    bool bcast_native = false;               /* bcast: KUPL native (A/B only) */
    bool ar_hier = false;                    /* KUCCL_KUPL_AR_ALGO=hier: layered at any size */
    bool ar_auto = false;                    /* KUCCL_KUPL_AR_ALGO=auto: size-routed (hier ≥ min) */
    uint64_t ar_hier_min = 512 * 1024;       /* auto's hier floor */
    bool a2a_direct = true;                  /* direct pull default; native kept for A/B */

    Config()
    {
        if (const char *e = std::getenv("KUCCL_KUPL_AG_ALGO")) {
            if (std::strcmp(e, "linear") == 0) {
                forced_algo = ALLGATHER_LINEAR; /* sync per KUCCL_KUPL_AG_SYNC */
            } else if (std::strcmp(e, "hier_die") == 0) {
                forced_algo = ALLGATHER_DIE_PAIR;
            } else if (std::strcmp(e, "hier_socket") == 0) {
                forced_algo = ALLGATHER_SOCKET_PAIR;
            } else if (std::strcmp(e, "auto") == 0 || std::strcmp(e, "hier") == 0) {
                topo_auto = true; /* topology-routed (unset default is linear) */
            } else if (std::strcmp(e, "smart") == 0) {
                smart_route = true; /* lazy learning on real calls */
            } else {
                /* Fail loud: a typo'd value silently falling back to default
                 * routing corrupts A/B comparisons. Runs at .so load, before
                 * the log system is up. */
                std::fprintf(stderr,
                             "[KuCCL][WARN] KUCCL_KUPL_AG_ALGO='%s' unrecognized (valid: auto linear hier "
                             "hier_die hier_socket smart) — defaulting to linear\n",
                             e);
            }
        }
        if (const char *e = std::getenv("KUCCL_KUPL_AG_SYNC")) {
            if (std::strcmp(e, "doorbell") == 0) {
                sync_doorbell = true;
            } else if (std::strcmp(e, "fence") == 0) {
                sync_doorbell = false;
            } else {
                std::fprintf(stderr,
                             "[KuCCL][WARN] KUCCL_KUPL_AG_SYNC='%s' unrecognized (valid: fence doorbell) — "
                             "using fence\n",
                             e);
            }
        }
        if (const char *e = std::getenv("KUCCL_KUPL_AR_HIER_MIN"))
            ar_hier_min = parse_bytes(e, ar_hier_min);
        if (const char *e = std::getenv("KUCCL_KUPL_A2A_ALGO")) {
            if (std::strcmp(e, "direct") == 0 || std::strcmp(e, "auto") == 0) {
                a2a_direct = true;
            } else if (std::strcmp(e, "native") == 0) {
                a2a_direct = false; /* KUPL native alltoall (A/B only) */
                /* Keep the existing preference for linear_read in native A/B
                 * runs; respect an explicit user override. Verification must
                 * determine correctness for the installed KUPL version. */
                setenv("KUPL_SHM_ALLTOALL_ALGORITHM", "1", 0);
            } else {
                std::fprintf(stderr,
                             "[KuCCL][WARN] KUCCL_KUPL_A2A_ALGO='%s' unrecognized (valid: auto native direct) — "
                             "using native\n",
                             e);
            }
        }
        if (const char *e = std::getenv("KUCCL_KUPL_PEER_MAP")) {
            if (std::strcmp(e, "kupl") == 0) {
                peer_map_own = false; /* legacy KUPL attach (self-alias defect on some builds) */
            } else if (std::strcmp(e, "own") != 0) {
                std::fprintf(stderr,
                             "[KuCCL][WARN] KUCCL_KUPL_PEER_MAP='%s' unrecognized (valid: own kupl) — using own\n", e);
            }
        }
        if (const char *e = std::getenv("KUCCL_KUPL_SMART_REPS")) {
            const long v = std::strtol(e, nullptr, 10);
            if (v >= 1 && v <= 10)
                smart_reps = static_cast<int>(v);
        }
        if (const char *e = std::getenv("KUCCL_KUPL_AG_THREADS")) {
            const long v = std::strtol(e, nullptr, 10);
            if (v >= 1 && v <= 64)
                parallel_threads = static_cast<int>(v);
        }
        if (const char *e = std::getenv("KUCCL_KUPL_AG_PAR_MIN"))
            parallel_min_bytes = parse_bytes(e, parallel_min_bytes);
        if (const char *e = std::getenv("KUCCL_KUPL_PF_DIST")) {
            const long v = std::strtol(e, nullptr, 10);
            if (v >= 0)
                pf_dist = static_cast<int>(v);
        }
        if (const char *e = std::getenv("KUCCL_KUPL_SPIN_MAX")) {
            const unsigned long long v = std::strtoull(e, nullptr, 10);
            if (v > 0)
                spin_max = v;
        }
        if (const char *e = std::getenv("KUCCL_KUPL_SCRATCH_INIT")) {
            const unsigned long long v = std::strtoull(e, nullptr, 10);
            if (v > 0)
                scratch_floor = v;
        }
        if (const char *e = std::getenv("KUCCL_KUPL_BC_ALGO"))
            bcast_native = std::strcmp(e, "native") == 0;
        if (const char *e = std::getenv("KUCCL_KUPL_AR_ALGO")) {
            if (std::strcmp(e, "hier") == 0) {
                ar_hier = true; /* layered at any size (see ar_route) */
            } else if (std::strcmp(e, "auto") == 0) {
                ar_auto = true; /* size-routed: hier past ar_hier_min, flat below */
            } else if (std::strcmp(e, "flat") == 0) {
                /* flat at every size (same as unset) */
            } else {
                std::fprintf(stderr,
                             "[KuCCL][WARN] KUCCL_KUPL_AR_ALGO='%s' unrecognized (valid: auto flat hier) — "
                             "defaulting to flat\n",
                             e);
            }
        }
    }

    /* Unset default: linear direct reads. Explicit "auto"/"hier": layer
     * where the topology allows it — socket first, die second, flat →
     * linear. linear / hier_die / hier_socket override for A/B (unsuitable
     * granularity demotes). */
    AllgatherAlgo route(bool die_ok, bool socket_ok) const
    {
        if (forced_algo == ALLGATHER_LINEAR)
            return ALLGATHER_LINEAR;
        if (forced_algo == ALLGATHER_DIE_PAIR)
            return die_ok ? ALLGATHER_DIE_PAIR : ALLGATHER_LINEAR;
        if (forced_algo == ALLGATHER_SOCKET_PAIR)
            return socket_ok ? ALLGATHER_SOCKET_PAIR : (die_ok ? ALLGATHER_DIE_PAIR : ALLGATHER_LINEAR);
        if (topo_auto) {
            if (socket_ok)
                return ALLGATHER_SOCKET_PAIR;
            if (die_ok)
                return ALLGATHER_DIE_PAIR;
        }
        return ALLGATHER_LINEAR;
    }

    /* AllReduce routing — flat (default), hier (layered at any size), auto
     * (size-routed: hier past ar_hier_min, flat below). */
    AllreduceAlgo ar_route(size_t bytes, bool die_ok, bool socket_ok) const
    {
        if (ar_hier || (ar_auto && bytes >= ar_hier_min)) {
            if (socket_ok)
                return ALLREDUCE_HIER_SOCKET;
            if (die_ok)
                return ALLREDUCE_HIER_DIE;
        }
        return ALLREDUCE_FLAT;
    }

private:
    static uint64_t parse_bytes(const char *s, uint64_t dflt)
    {
        const unsigned long long v = std::strtoull(s, nullptr, 10);
        return v > 0 ? v : dflt;
    }
};

const Config g_config;
/* Route name for logs: structure + (for linear) the sync primitive. */
static const char *ag_name(uint8_t algo, bool doorbell)
{
    switch (static_cast<AllgatherAlgo>(algo)) {
        case ALLGATHER_DIE_PAIR:
            return "hier_die";
        case ALLGATHER_SOCKET_PAIR:
            return "hier_socket";
        default:
            return doorbell ? "linear(doorbell)" : "linear(fence)";
    }
}

/* Route name for logs. */
static const char *ar_name(uint8_t algo)
{
    switch (static_cast<AllreduceAlgo>(algo)) {
        case ALLREDUCE_HIER_DIE:
            return "hier_die";
        case ALLREDUCE_HIER_SOCKET:
            return "hier_socket";
        default:
            return "flat";
    }
}

/* ── Timing (KUCCL_KUPL_TIMING=1) ── */

enum TimingPhase {
    PHASE_SCRATCH,
    PHASE_STAGE_OUT,
    PHASE_FENCE,
    PHASE_FLAG_WAIT,
    PHASE_ACK_WAIT,
    PHASE_DATA_MOVE,
    PHASE_NATIVE,
    PHASE_COPY_BACK,
    PHASE_COUNT
};
enum OpKind {
    OP_ALLGATHER,
    OP_ALLREDUCE,
    OP_BROADCAST,
    OP_ALLTOALL,
    OP_COUNT
};

struct KuplTiming {
    uint64_t ns[PHASE_COUNT] = {};
    uint64_t cnt[PHASE_COUNT] = {};
    uint64_t ns_op[OP_COUNT] = {};
    uint64_t cnt_op[OP_COUNT] = {};
    uint64_t fence_cnt = 0;
    static uint64_t now()
    {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
    }
};

KuplTiming g_timing;

/* allgather latency bucketed per chunk size (averages across 8B..1MB are
 * meaningless; buckets align with benchmark tables) */
#define MAX_LATENCY_BUCKETS 64
struct LatencyBucket {
    uint64_t chunk;
    uint64_t calls;
    uint64_t total_ns;
};
static LatencyBucket g_latency_buckets[MAX_LATENCY_BUCKETS];
static int g_num_latency_buckets = 0;

static void record_allgather_latency(uint64_t chunk, uint64_t elapsed_ns)
{
    for (int i = 0; i < g_num_latency_buckets; i++) {
        if (g_latency_buckets[i].chunk == chunk) {
            g_latency_buckets[i].calls++;
            g_latency_buckets[i].total_ns += elapsed_ns;
            return;
        }
    }
    if (g_num_latency_buckets < MAX_LATENCY_BUCKETS) {
        g_latency_buckets[g_num_latency_buckets] = {chunk, 1, elapsed_ns};
        g_num_latency_buckets++;
    }
}

inline bool timing_enabled()
{
    static const bool v = [] {
        const char *e = std::getenv("KUCCL_KUPL_TIMING");
        return e != nullptr && e[0] != '\0' && std::strcmp(e, "0") != 0;
    }();
    return v;
}

/* RAII scope: near-zero cost when off (single branch) */
struct TimingScope {
    TimingScope(bool on, uint64_t &ns, uint64_t &cnt) : on_(on), ns_(&ns), cnt_(&cnt), t0_(on ? KuplTiming::now() : 0)
    {
    }
    ~TimingScope()
    {
        if (on_) {
            *ns_ += KuplTiming::now() - t0_;
            (*cnt_)++;
        }
    }
    bool on_;
    uint64_t *ns_;
    uint64_t *cnt_;
    uint64_t t0_;
};

/* fence wrapper: accumulated + per-call TRACE */
inline void timed_fence(kupl_shm_win_h win, int rank)
{
    const bool on = timing_enabled();
    const uint64_t t0 = on ? KuplTiming::now() : 0;
    kupl_shm_fence(win);
    if (on) {
        const uint64_t d = KuplTiming::now() - t0;
        g_timing.ns[PHASE_FENCE] += d;
        g_timing.cnt[PHASE_FENCE]++;
        g_timing.fence_cnt++;
        KuCCL_LOG_TRACE(rank, "[TIMING] fence#%llu: %.1f us", static_cast<unsigned long long>(g_timing.fence_cnt),
                        static_cast<double>(d) / 1000.0);
    }
}

/* Summary at comm destroy (called from kupl_group_fini); KUCCL_KUPL_TIMING=1
 * is the switch. */
static void kupl_timing_report(int rank, int nranks)
{
    if (!timing_enabled())
        return;
    static const char *ph[] = {"scratch",  "stage_out", "fence",       "flag_wait",
                               "ack_wait", "data_read", "kupl_native", "copy_back"};
    static const char *opn[] = {"allgather", "allreduce", "broadcast", "alltoall"};
    detail::log_write(detail::LogLevel::INFO, rank, __FILE__, __LINE__,
                      "[TIMING] summary ws=%d (KUCCL_KUPL_TIMING=1; cumulative across comms in this process; "
                      "phases sorted by total time)",
                      nranks);
    detail::log_write(detail::LogLevel::INFO, rank, __FILE__, __LINE__,
                      "[TIMING] legend scratch=window mgmt stage_out=data into window fence=full barrier "
                      "flag_wait=doorbell wait ack_wait=drain wait data_read=pure data move kupl_native=KUPL op "
                      "copy_back=result to user");
    int order[PHASE_COUNT];
    int n_order = 0;
    for (int i = 0; i < PHASE_COUNT; i++)
        if (g_timing.cnt[i] != 0)
            order[n_order++] = i;
    for (int a = 0; a < n_order - 1; a++)
        for (int b = 0; b < n_order - 1 - a; b++)
            if (g_timing.ns[order[b]] < g_timing.ns[order[b + 1]]) {
                const int t = order[b];
                order[b] = order[b + 1];
                order[b + 1] = t;
            }
    for (int k = 0; k < n_order; k++) {
        const int i = order[k];
        detail::log_write(detail::LogLevel::INFO, rank, __FILE__, __LINE__,
                          "[TIMING] %-11s %14.1f us / %6llu times "
                          "(avg %9.1f us)",
                          ph[i], static_cast<double>(g_timing.ns[i]) / 1000.0,
                          static_cast<unsigned long long>(g_timing.cnt[i]),
                          static_cast<double>(g_timing.ns[i]) / 1000.0 / static_cast<double>(g_timing.cnt[i]));
    }
    if (g_timing.fence_cnt != 0)
        detail::log_write(detail::LogLevel::INFO, rank, __FILE__, __LINE__,
                          "[TIMING] explicit kupl_shm_fence total %llu calls (per-call detail in TRACE log)",
                          static_cast<unsigned long long>(g_timing.fence_cnt));
    for (int i = 0; i < OP_COUNT; i++) {
        if (g_timing.cnt_op[i] == 0)
            continue;
        detail::log_write(detail::LogLevel::INFO, rank, __FILE__, __LINE__,
                          "[TIMING] [op:%s] %14.1f us / %6llu times "
                          "(avg %9.1f us)",
                          opn[i], static_cast<double>(g_timing.ns_op[i]) / 1000.0,
                          static_cast<unsigned long long>(g_timing.cnt_op[i]),
                          static_cast<double>(g_timing.ns_op[i]) / 1000.0 / static_cast<double>(g_timing.cnt_op[i]));
    }
    if (g_timing.cnt_op[OP_ALLREDUCE] > 0 && g_timing.cnt_op[OP_ALLGATHER] > 0)
        detail::log_write(detail::LogLevel::INFO, rank, __FILE__, __LINE__,
                          "[TIMING] note: allreduce calls come from harness dist.barrier() between size steps — "
                          "outside the allgather timing window");
    if (g_num_latency_buckets > 0) {
        detail::log_write(detail::LogLevel::INFO, rank, __FILE__, __LINE__,
                          "[TIMING] ── allgather by chunk size (avg latency per bucket) ──");
        for (int i = 0; i < g_num_latency_buckets; i++)
            detail::log_write(detail::LogLevel::INFO, rank, __FILE__, __LINE__,
                              "[TIMING] [size] %10lluB  avg %9.1f us"
                              "  × %3llu times",
                              static_cast<unsigned long long>(g_latency_buckets[i].chunk),
                              static_cast<double>(g_latency_buckets[i].total_ns) / 1000.0 /
                                  static_cast<double>(g_latency_buckets[i].calls),
                              static_cast<unsigned long long>(g_latency_buckets[i].calls));
    }
}

/* kucclDataType_t → kupl_shm_datatype_t. Signed/unsigned pairs share the
 * underlying type: SUM/MIN/MAX are bit-identical in two's complement. */
kupl_shm_datatype_t map_datatype(kucclDataType_t dt)
{
    switch (dt) {
        case kucclInt8:
        case kucclUint8:
            return KUPL_SHM_DATATYPE_CHAR;
        case kucclInt32:
        case kucclUint32:
            return KUPL_SHM_DATATYPE_INT;
        case kucclInt64:
        case kucclUint64:
            return KUPL_SHM_DATATYPE_LONG;
        case kucclFloat:
            return KUPL_SHM_DATATYPE_FLOAT;
        case kucclDouble:
            return KUPL_SHM_DATATYPE_DOUBLE;
    }
    return KUPL_SHM_DATATYPE_CHAR;
}

kupl_shm_reduce_op_t map_redop(kucclRedOp_t op)
{
    switch (op) {
        case kucclSum:
            return KUPL_SHM_REDUCE_OP_SUM;
        case kucclMin:
            return KUPL_SHM_REDUCE_OP_MIN;
        case kucclMax:
            return KUPL_SHM_REDUCE_OP_MAX;
    }
    return KUPL_SHM_REDUCE_OP_SUM;
}

size_t kupl_dt_size(kucclDataType_t dt)
{
    switch (dt) {
        case kucclInt8:
        case kucclUint8:
            return 1;
        case kucclInt32:
        case kucclUint32:
        case kucclFloat:
            return 4;
        case kucclInt64:
        case kucclUint64:
        case kucclDouble:
            return 8;
    }
    return 0; /* unknown datatype — rejected at dispatch */
}

/* Element-wise MIN/MAX (AllReduce non-SUM paths). Keep the element loops
 * clean — bisheng clang auto-vectorizes them (SVE). Remote-range prefetch
 * happens line-granular in dispatch_* BEFORE the kernel (see
 * prefetch_peer_range): a conditional prefetch inside the loop would break
 * auto-vectorization and collapse throughput to scalar. */
template <typename T>
static void minmax_op(void *result, const void *peer, int count, bool is_min)
{
    auto *r = static_cast<T *>(result);
    const auto *p = static_cast<const T *>(peer);
    if (is_min) {
        for (int i = 0; i < count; i++)
            if (p[i] < r[i])
                r[i] = p[i];
    } else {
        for (int i = 0; i < count; i++)
            if (p[i] > r[i])
                r[i] = p[i];
    }
}

/* Line-granular prefetch of a peer range, one PRFM per 64B line,
 * KUCCL_KUPL_PF_DIST ahead. Kept out of the element kernels so they stay
 * auto-vectorizable; ranges beyond L2 skip it (stream prefetcher territory). */
static void prefetch_peer_range(const void *peer, size_t bytes)
{
    const size_t pf = static_cast<size_t>(g_config.pf_dist);
    if (pf == 0 || bytes <= pf || bytes > 512 * 1024)
        return;
    const char *p = static_cast<const char *>(peer);
    for (size_t off = 0; off + pf < bytes; off += 64)
        __builtin_prefetch(p + off + pf, 0, 0);
}

static void dispatch_minmax(kucclDataType_t dt, kucclRedOp_t op, void *result, const void *peer, int count)
{
    prefetch_peer_range(peer, static_cast<size_t>(count) * kupl_dt_size(dt));
    const bool is_min = (op == kucclMin);
    switch (dt) {
        case kucclInt8:
            minmax_op<int8_t>(result, peer, count, is_min);
            break;
        case kucclUint8:
            minmax_op<uint8_t>(result, peer, count, is_min);
            break;
        case kucclInt32:
            minmax_op<int32_t>(result, peer, count, is_min);
            break;
        case kucclUint32:
            minmax_op<uint32_t>(result, peer, count, is_min);
            break;
        case kucclInt64:
            minmax_op<int64_t>(result, peer, count, is_min);
            break;
        case kucclUint64:
            minmax_op<uint64_t>(result, peer, count, is_min);
            break;
        case kucclFloat:
            minmax_op<float>(result, peer, count, is_min);
            break;
        case kucclDouble:
            minmax_op<double>(result, peer, count, is_min);
            break;
    }
}

/* Element-wise SUM (layered AllReduce path; flat SUM rides KUPL native).
 * Clean loop on purpose: auto-vectorized by the compiler. */
template <typename T>
static void sum_op(void *result, const void *peer, int count)
{
    auto *r = static_cast<T *>(result);
    const auto *p = static_cast<const T *>(peer);
    for (int i = 0; i < count; i++)
        r[i] = r[i] + p[i];
}

static void dispatch_sum(kucclDataType_t dt, void *result, const void *peer, int count)
{
    prefetch_peer_range(peer, static_cast<size_t>(count) * kupl_dt_size(dt));
    switch (dt) {
        case kucclInt8:
            sum_op<int8_t>(result, peer, count);
            break;
        case kucclUint8:
            sum_op<uint8_t>(result, peer, count);
            break;
        case kucclInt32:
            sum_op<int32_t>(result, peer, count);
            break;
        case kucclUint32:
            sum_op<uint32_t>(result, peer, count);
            break;
        case kucclInt64:
            sum_op<int64_t>(result, peer, count);
            break;
        case kucclUint64:
            sum_op<uint64_t>(result, peer, count);
            break;
        case kucclFloat:
            sum_op<float>(result, peer, count);
            break;
        case kucclDouble:
            sum_op<double>(result, peer, count);
            break;
    }
}

/* SUM/MIN/MAX combine dispatch (layered AllReduce). */
static void dispatch_combine(kucclDataType_t dt, kucclRedOp_t op, void *result, const void *peer, int count)
{
    if (op == kucclSum)
        dispatch_sum(dt, result, peer, count);
    else
        dispatch_minmax(dt, op, result, peer, count);
}

/* Recursive halving/doubling range: [start, end] owned by rank after step.
 * Each step halves against partner (rank ^ mask); lower rank keeps the left
 * half. Mirrors the KUPL SUM algorithm's partitioning. */
static void calc_range(int rank, int count, int step, int *start, int *end)
{
    *start = 0;
    *end = count - 1;
    for (int i = 0; i <= step; i++) {
        const int mask = 1 << i;
        const int partner = rank ^ mask;
        const int mid = (*start + *end) / 2;
        if (rank < partner)
            *end = mid;
        else
            *start = mid + 1;
    }
}

static inline bool is_pow2(int n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

static inline size_t align_page(size_t n)
{
    const size_t page = 4096;
    return (n + page - 1) / page * page;
}

/* "0-3,8,10-11" cpulist membership */
static bool cpu_in_list(const char *s, int cpu)
{
    const char *p = s;
    while (*p != '\0' && *p != '\n') {
        char *end = nullptr;
        const long lo = std::strtol(p, &end, 10);
        if (end == p)
            break;
        long hi = lo;
        p = end;
        if (*p == '-') {
            hi = std::strtol(p + 1, &end, 10);
            p = end;
        }
        if (cpu >= lo && cpu <= hi)
            return true;
        while (*p == ',' || *p == ' ' || *p == '\t')
            p++;
    }
    return false;
}

static int detect_numa_node()
{
#if defined(__linux__)
    unsigned cpu = 0, node = 0;
    if (::syscall(SYS_getcpu, &cpu, &node, nullptr) == 0)
        return static_cast<int>(node);
    const int cur = sched_getcpu();
    char path[96];
    char buf[512];
    for (int n = 0; n < 4096; n++) {
        std::snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", n);
        std::FILE *f = std::fopen(path, "r");
        if (f == nullptr)
            continue; /* sparse numbering: skip, don't stop */
        const bool ok = std::fgets(buf, sizeof(buf), f) != nullptr;
        std::fclose(f);
        if (ok && cpu_in_list(buf, cur))
            return n;
    }
#endif
    return -1;
}

static int detect_socket_id()
{
#if defined(__linux__)
    const int cpu = sched_getcpu();
    char path[128];
    char buf[32];
    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
    std::FILE *f = std::fopen(path, "r");
    if (f == nullptr)
        return -1;
    const bool ok = std::fgets(buf, sizeof(buf), f) != nullptr;
    std::fclose(f);
    if (!ok)
        return -1;
    return static_cast<int>(std::strtol(buf, nullptr, 10));
#else
    return -1;
#endif
}

struct Topo {
    /* ── die granularity (NUMA node) ── */
    bool usable = false;                  /* ≥2 nodes, ≥2 ranks/node (uneven OK: relaxed pairing) */
    int my_node = -1;                     /* compressed node index */
    std::vector<int> node_of;             /* rank → compressed node index */
    std::vector<int> local_ranks;         /* ranks on my node (ascending, incl. self) */
    std::vector<int> partners;            /* per remote node: my counterpart rank */
    std::vector<int> remote_ranks;        /* all ranks outside my node (ascending) */
    std::vector<std::vector<int>> groups; /* all groups' member lists (die_relay) */
    std::vector<int> leaders;             /* per group: lowest rank = relay head */

    /* ── socket granularity (physical package) — hier_socket's input ── */
    bool socket_usable = false;                  /* ≥2 sockets, ≥2 ranks/socket (uneven OK: relaxed pairing) */
    int my_socket = -1;                          /* compressed socket index */
    std::vector<int> socket_local_ranks;         /* ranks on my socket (ascending, incl. self) */
    std::vector<int> socket_partners;            /* per remote socket: my counterpart rank */
    std::vector<int> socket_remote_ranks;        /* all ranks outside my socket (ascending) */
    std::vector<std::vector<int>> socket_groups; /* all socket groups (layered AllReduce slice sizing) */
    std::vector<int> rev_partners;               /* die granularity: ranks whose partner list contains me
                                            (doorbell signal targets — partner relation is NOT
                                            symmetric under relaxed clamping) */
    std::vector<int> socket_rev_partners;        /* socket granularity: same, socket level */
};

/* Distribution summary for diagnostics: "node0:8 node1:8" */
static void topo_dist(char *buf, size_t cap, const char *label, const std::vector<int> &ids,
                      const std::vector<std::vector<int>> &groups)
{
    size_t off = 0;
    for (size_t j = 0; j < groups.size() && off + 1 < cap; j++)
        off += static_cast<size_t>(
            std::snprintf(buf + off, cap - off, "%s%s%d:%zu", j > 0 ? " " : "", label, ids[j], groups[j].size()));
}

/* Topology build; every early return logs its reason. */
static void topo_build(Topo &topo, const std::vector<int> &node_of, int rank)
{
    const int n = static_cast<int>(node_of.size());
    for (int i = 0; i < n; i++) {
        if (node_of[i] < 0)
            return;
    }
    std::vector<int> ids(node_of);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    if (ids.size() < 2)
        return;

    std::vector<std::vector<int>> groups(ids.size());
    for (int i = 0; i < n; i++) {
        const size_t g = static_cast<size_t>(std::lower_bound(ids.begin(), ids.end(), node_of[i]) - ids.begin());
        groups[g].push_back(i);
    }
    for (const auto &g : groups) {
        if (g.size() < 2) {
            char dist[128];
            topo_dist(dist, sizeof(dist), "node", ids, groups);
            KuCCL_LOG_INFO(rank, "kupl topo: flat — a die holds fewer than 2 ranks (%s)", dist);
            return;
        }
    }

    topo.node_of = node_of;
    const size_t my_g = static_cast<size_t>(std::lower_bound(ids.begin(), ids.end(), node_of[rank]) - ids.begin());
    topo.my_node = static_cast<int>(my_g);
    topo.local_ranks = groups[my_g];
    const int local_idx = static_cast<int>(std::lower_bound(topo.local_ranks.begin(), topo.local_ranks.end(), rank) -
                                           topo.local_ranks.begin());
    /* Relaxed pairing for uneven spreads: ranks beyond a remote group's size
     * clamp onto its last member — the duplicate read lands the same bytes in
     * the same slot, so concurrent writes are benign. */
    for (size_t j = 0; j < groups.size(); j++) {
        if (j == my_g)
            continue;
        topo.partners.push_back(groups[j][std::min<size_t>(local_idx, groups[j].size() - 1)]);
        topo.remote_ranks.insert(topo.remote_ranks.end(), groups[j].begin(), groups[j].end());
    }
    std::sort(topo.remote_ranks.begin(), topo.remote_ranks.end());
    topo.usable = true;
    /* Reverse partner relation: who waits on my flags. w's partner in my group
     * is groups[my_g][min(idx_w, my_size-1)] — clamping makes this asymmetric
     * with topo.partners, so doorbell producers must signal this set. */
    for (size_t j = 0; j < groups.size(); j++) {
        if (j == my_g)
            continue;
        for (const int w : groups[j]) {
            const size_t iw =
                static_cast<size_t>(std::lower_bound(groups[j].begin(), groups[j].end(), w) - groups[j].begin());
            if (groups[my_g][std::min(iw, groups[my_g].size() - 1)] == rank)
                topo.rev_partners.push_back(w);
        }
    }
    std::sort(topo.rev_partners.begin(), topo.rev_partners.end());
    topo.groups = groups;
    for (const auto &g : groups)
        topo.leaders.push_back(g[0]);
    char dist[128];
    topo_dist(dist, sizeof(dist), "node", ids, groups);
    KuCCL_LOG_INFO(rank, "kupl topo: hier_die enabled (%s, %d remote partner(s))", dist,
                   static_cast<int>(topo.partners.size()));
}

static void socket_build(Topo &topo, const std::vector<int> &socket_of, int rank)
{
    const int n = static_cast<int>(socket_of.size());
    for (int i = 0; i < n; i++) {
        if (socket_of[i] < 0)
            return;
    }
    std::vector<int> ids(socket_of);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    if (ids.size() < 2) {
        KuCCL_LOG_INFO(rank,
                       "kupl topo: socket_pair off — all %d ranks on one socket (die-granular topo covers this "
                       "layout)",
                       n);
        return;
    }

    std::vector<std::vector<int>> groups(ids.size());
    for (int i = 0; i < n; i++) {
        const size_t g = static_cast<size_t>(std::lower_bound(ids.begin(), ids.end(), socket_of[i]) - ids.begin());
        groups[g].push_back(i);
    }
    for (const auto &g : groups) {
        if (g.size() < 2) {
            char dist[128];
            topo_dist(dist, sizeof(dist), "socket", ids, groups);
            KuCCL_LOG_INFO(rank, "kupl topo: socket_pair off — a socket holds fewer than 2 ranks (%s)", dist);
            return;
        }
    }

    const size_t my_g = static_cast<size_t>(std::lower_bound(ids.begin(), ids.end(), socket_of[rank]) - ids.begin());
    topo.my_socket = static_cast<int>(my_g);
    topo.socket_local_ranks = groups[my_g];
    const int local_idx =
        static_cast<int>(std::lower_bound(topo.socket_local_ranks.begin(), topo.socket_local_ranks.end(), rank) -
                         topo.socket_local_ranks.begin());
    /* Same relaxed pairing as topo_build: clamp onto the remote group's last
     * member when spreads are uneven. */
    for (size_t j = 0; j < groups.size(); j++) {
        if (j == my_g)
            continue;
        topo.socket_partners.push_back(groups[j][std::min<size_t>(local_idx, groups[j].size() - 1)]);
        topo.socket_remote_ranks.insert(topo.socket_remote_ranks.end(), groups[j].begin(), groups[j].end());
    }
    std::sort(topo.socket_remote_ranks.begin(), topo.socket_remote_ranks.end());
    topo.socket_usable = true;
    for (size_t j = 0; j < groups.size(); j++) {
        if (j == my_g)
            continue;
        for (const int w : groups[j]) {
            const size_t iw =
                static_cast<size_t>(std::lower_bound(groups[j].begin(), groups[j].end(), w) - groups[j].begin());
            if (groups[my_g][std::min(iw, groups[my_g].size() - 1)] == rank)
                topo.socket_rev_partners.push_back(w);
        }
    }
    std::sort(topo.socket_rev_partners.begin(), topo.socket_rev_partners.end());
    topo.socket_groups = groups; /* layered AllReduce: min group size must match on every rank */
    char dist[128];
    topo_dist(dist, sizeof(dist), "socket", ids, groups);
    KuCCL_LOG_INFO(rank, "kupl topo: socket_pair enabled (%s, %d remote socket partner(s))", dist,
                   static_cast<int>(topo.socket_partners.size()));
}

/* Smart routing — a lazy bandit. The first call in a size bucket serves the
 * static router's choice; repeated calls explore the remaining candidates on
 * real user calls; a worst-rank vote then fixes the winner.
 *
 * Safety invariants (routing divergence across ranks = protocol mismatch =
 * deadlock): per-bucket state evolves identically on every rank (sample
 * counts advance even on failed calls); the vote reads shared data via a
 * worst-rank max, so the argmin is identical everywhere; every path drains
 * before staging, so interleaving candidates is safe. */
struct SmartRoute {
    static constexpr int NUM_BUCKETS = 26; /* log2(chunk) clamped: 1B..32MB+ */
    static constexpr int MAX_CAND = 4;     /* linear-fence, linear-doorbell, hier_die, hier_socket */
    bool cand_built = false;
    int cand_total = 0;
    AgChoice cand_map[MAX_CAND] = {};
    uint8_t phase[NUM_BUCKETS] = {};              /* 0=unseen 1=exploring 2=learned */
    uint16_t samples[NUM_BUCKETS][MAX_CAND] = {}; /* per-candidate sample count */
    uint64_t sum_ns[NUM_BUCKETS][MAX_CAND] = {};  /* per-candidate latency sum */
    AgChoice choice[NUM_BUCKETS] = {};            /* static (phase 1) or learned (2) */
};

/* Per-comm private state (kucclComm::be_ctx). Peer segment bases are cached
 * at window (re)allocation — hot paths never call win_query. */
struct KuplComm {
    kucclComm *owner = nullptr; /* back-pointer (OOB callback cookie) */
    kupl_shm_comm_h comm = nullptr;
    int nranks = 0;

    /* Staging windows. send_win and recv_win MUST stay separate windows:
     * KUPL native algorithms select send/recv regions by window handle
     * (find_win), a shared window would misroute reads. */
    kupl_shm_win_h send_win = nullptr;
    kupl_shm_win_h recv_win = nullptr;
    char *send_base = nullptr;
    char *recv_base = nullptr;
    size_t send_cap = 0; /* bytes per rank segment */
    size_t recv_cap = 0;
    std::vector<char *> send_ptrs; /* [nranks] peer send segment bases */
    std::vector<char *> recv_ptrs; /* [nranks] peer recv segment bases */

    kupl_shm_win_h notify_win = nullptr;
    char *notify_base = nullptr;
    volatile int64_t *ready_flag = nullptr;
    volatile int64_t *ack_flag = nullptr;
    std::vector<volatile int64_t *> peer_ready; /* [nranks] */
    std::vector<volatile int64_t *> peer_ack;   /* [nranks] */

    uint64_t flag_round = 0; /* flag-protocol round (allgather + bcast share it) */
    uint64_t db_seq = 0;     /* hier-path doorbell sequence (pair_hybrid + layered AR) */
    Topo topo;
    SmartRoute smart;
    bool broken = false; /* window cache failed — fail fast on further use */
};

/* Per-request private state (kucclRequest::priv) */
struct KuplReq {
    kupl_shm_request_h req = nullptr;
    void *user_dst = nullptr;      /* copy-back destination (nullptr = none) */
    const void *win_src = nullptr; /* copy-back source (staging recv region) */
    size_t copy_bytes = 0;
};

/* OOB adapters: kuccl callbacks (config.oob_*) → kupl signatures */
int kupl_oob_allgather_cb(const void *sendbuf, void *recvbuf, int size, void *group, kupl_shm_datatype_t)
{
    auto *ctx = static_cast<KuplComm *>(group);
    if (ctx == nullptr || ctx->owner == nullptr)
        return -1;
    return detail::oob_exchange(&ctx->owner->oob_config, sendbuf, recvbuf, size);
}

/* OOB barrier via 1-byte OOB allgather (semantics: everyone arrived) */
int kupl_oob_barrier_cb(void *group)
{
    auto *ctx = static_cast<KuplComm *>(group);
    if (ctx == nullptr || ctx->owner == nullptr)
        return -1;
    const char send_byte = 0;
    std::vector<char> recv(static_cast<size_t>(ctx->owner->nranks), 1);
    return detail::oob_exchange(&ctx->owner->oob_config, &send_byte, recv.data(), 1);
}

/* Extract the shm FILE backing `base` from /proc/self/maps. Returns false
 * for anonymous or unreadable mappings — the caller treats that uniformly
 * across ranks (see cache_peer_bases). */
static bool peer_map_filename(const void *base, char *out, size_t cap)
{
    FILE *fp = std::fopen("/proc/self/maps", "r");
    if (fp == nullptr)
        return false;
    char line[1024];
    const uintptr_t addr = reinterpret_cast<uintptr_t>(base);
    bool found = false;
    while (!found && std::fgets(line, sizeof(line), fp) != nullptr) {
        uintptr_t lo = 0, hi = 0;
        if (std::sscanf(line, "%lx-%lx", &lo, &hi) != 2 || addr < lo || addr >= hi)
            continue;
        /* skip 5 fields (addr perms offset dev inode); path is field 6 */
        const char *p = line;
        for (int field = 0; field < 5 && *p != '\0'; field++) {
            while (*p == ' ' || *p == '\t')
                p++;
            while (*p != ' ' && *p != '\t' && *p != '\0')
                p++;
        }
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '/') {
            std::snprintf(out, cap, "%s", p);
            for (char *nl = out; *nl != '\0'; nl++)
                if (*nl == '\n')
                    *nl = '\0';
            found = true;
        }
    }
    std::fclose(fp);
    return found;
}

/* Peer-mapping establishment. KUPL's window attach can map peer slots to
 * the rank's own segment on some builds, so the self-built path is the
 * default: exchange segment filenames over kuccl's OOB and mmap the peers
 * ourselves. kupl fences are internal to the window and unaffected.
 *
 * The branch depends only on g_config.peer_map_own (process-lifetime env);
 * every per-rank failure after the exchange is fatal — no rank may diverge
 * into the win_query path while others sit in the OOB allgather. */
static kucclResult_t self_peer_map(KuplComm *ctx, const char *own_base, size_t seg_bytes, std::vector<char *> &out,
                                   const char *name)
{
    const int nranks = ctx->nranks;
    out.assign(static_cast<size_t>(nranks), nullptr);
    constexpr int FNAME = 256;
    char my_file[FNAME] = {};
    peer_map_filename(own_base, my_file, FNAME); /* failure handled post-exchange */
    std::vector<char> all(static_cast<size_t>(nranks) * FNAME, 0);
    if (detail::oob_exchange(&ctx->owner->oob_config, my_file, all.data(), FNAME) != 0) {
        detail::err_record("kupl: %s filename OOB exchange FAILED", name);
        return KuCCL_FAIL;
    }
    if (my_file[0] != '/') {
        detail::err_record("kupl: %s own segment file not found in /proc/self/maps", name);
        return KuCCL_FAIL;
    }
    for (int peer = 0; peer < nranks; peer++) {
        if (peer == ctx->owner->rank) {
            out[static_cast<size_t>(peer)] = const_cast<char *>(own_base);
            continue;
        }
        const char *fname = all.data() + static_cast<size_t>(peer) * FNAME;
        if (fname[0] != '/') {
            detail::err_record("kupl: %s peer %d exchanged no filename", name, peer);
            return KuCCL_FAIL;
        }
        /* fname is the full path from /proc/self/maps — a plain filesystem
         * path, not a posix-shm *name*. open() it directly: shm_open would
         * prepend /dev/shm again ("/dev/shm/dev/shm/...") and miss. */
        const int fd = open(fname, O_RDWR);
        if (fd < 0) {
            detail::err_record("kupl: %s open(%s, peer=%d) FAILED errno=%d", name, fname, peer, errno);
            return KuCCL_FAIL;
        }
        void *seg = mmap(nullptr, seg_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (seg == MAP_FAILED) {
            detail::err_record("kupl: %s mmap(%s, peer=%d, %zuB) FAILED errno=%d", name, fname, peer, seg_bytes, errno);
            return KuCCL_FAIL;
        }
        out[static_cast<size_t>(peer)] = static_cast<char *>(seg);
    }
    const int sample = nranks > 1 ? 1 : 0;
    KuCCL_LOG_INFO(ctx->owner->rank, "kupl: %s peer map SELF-BUILT (own=%s p%d=%s)", name, my_file, sample,
                   all.data() + static_cast<size_t>(sample) * FNAME);
    return KuCCL_SUCCESS;
}

static kucclResult_t cache_peer_bases(KuplComm *ctx, kupl_shm_win_h win, const char *own_base, size_t seg_bytes,
                                      std::vector<char *> &out, const char *name)
{
    if (g_config.peer_map_own && ctx->owner != nullptr)
        return self_peer_map(ctx, own_base, seg_bytes, out, name);

    const int nranks = ctx->nranks;
    out.assign(static_cast<size_t>(nranks), nullptr);
    for (int peer = 0; peer < nranks; peer++) {
        void *base = nullptr;
        if (kupl_shm_win_query(win, peer, &base) != 0 || base == nullptr) {
            detail::err_record("kupl: win_query(%s, rank=%d) FAILED", name, peer);
            return KuCCL_FAIL;
        }
        out[peer] = static_cast<char *>(base);
    }
    return KuCCL_SUCCESS;
}

static kucclResult_t grow_win(KuplComm *ctx, kupl_shm_win_h *win, char **base, size_t *cap, std::vector<char *> *ptrs,
                              size_t bytes, const char *name)
{
    size_t needed = align_page(bytes);
    if (*win != nullptr && *cap >= needed)
        return KuCCL_SUCCESS;
    if (*cap == 0 && needed < g_config.scratch_floor)
        needed = g_config.scratch_floor; /* first alloc: avoid collective regrowth */
    if (*cap > 0 && *cap * 2 > needed)
        needed = *cap * 2; /* geometric growth */
    if (*win != nullptr) {
        /* drop our self-built peer mappings of the old window (the own
         * segment's mapping belongs to the window — win_free unmaps it) */
        if (g_config.peer_map_own)
            for (char *p : *ptrs)
                if (p != nullptr && p != *base)
                    munmap(p, *cap);
        kupl_shm_win_free(*win);
        *win = nullptr;
        *base = nullptr;
    }
    void *new_base = nullptr;
    kupl_shm_win_h new_win = nullptr;
    if (kupl_shm_win_alloc(needed, ctx->comm, &new_base, &new_win) != 0 || new_base == nullptr || new_win == nullptr) {
        detail::err_record("kupl: %s win_alloc(%zu) FAILED", name, needed);
        return KuCCL_FAIL;
    }
    *win = new_win;
    *base = static_cast<char *>(new_base);
    *cap = needed;
    if (cache_peer_bases(ctx, new_win, *base, needed, *ptrs, name) != KuCCL_SUCCESS)
        return KuCCL_FAIL; /* window valid but caches incomplete — comm unusable */
    return KuCCL_SUCCESS;
}

static inline kucclResult_t ensure_send(KuplComm *ctx, size_t bytes)
{
    if (grow_win(ctx, &ctx->send_win, &ctx->send_base, &ctx->send_cap, &ctx->send_ptrs, bytes, "send") !=
        KuCCL_SUCCESS) {
        ctx->broken = true; /* caches incomplete — fail fast from now on */
        return KuCCL_FAIL;
    }
    return KuCCL_SUCCESS;
}

static inline kucclResult_t ensure_recv(KuplComm *ctx, size_t bytes)
{
    if (grow_win(ctx, &ctx->recv_win, &ctx->recv_base, &ctx->recv_cap, &ctx->recv_ptrs, bytes, "recv") !=
        KuCCL_SUCCESS) {
        ctx->broken = true; /* caches incomplete — fail fast from now on */
        return KuCCL_FAIL;
    }
    return KuCCL_SUCCESS;
}

static inline void spin_wait(volatile int64_t *counter, uint64_t threshold)
{
    uint64_t spins = 0;
    while (static_cast<uint64_t>(__atomic_load_n(counter, __ATOMIC_ACQUIRE)) < threshold) {
        if (++spins < g_config.spin_max) {
#if defined(__aarch64__)
            __builtin_arm_yield();
#endif
        } else {
            sched_yield();
        }
    }
}

static void drain_peer_acks(kucclComm *comm, KuplComm *ctx)
{
    if (ctx->flag_round == 0)
        return;
    for (int peer = 0; peer < comm->nranks; peer++) {
        if (peer == comm->rank)
            continue;
        TimingScope tw(timing_enabled(), g_timing.ns[PHASE_ACK_WAIT], g_timing.cnt[PHASE_ACK_WAIT]);
        spin_wait(ctx->peer_ack[peer], ctx->flag_round);
    }
}

static void abort_flag_round(KuplComm *ctx, uint64_t round)
{
    __atomic_store_n(ctx->ready_flag, static_cast<int64_t>(round), __ATOMIC_RELEASE);
    __atomic_store_n(ctx->ack_flag, static_cast<int64_t>(round), __ATOMIC_RELEASE);
    ctx->flag_round = round;
}

/* ── hier-path doorbells ──
 * Write remote, spin local: the producer pays one remote store per waiter;
 * the waiter spins on its own memory. Transport: per-phase arrival arrays
 * in the notify window, indexed by producer rank, holding the seq of the
 * producer's latest completed phase. The seq bumps once per hier call
 * (before any failure return); stale slots are simply < seq — no flag
 * reset, no flip buffers.
 * Arrays are 128B-strided (one cache line per phase, no false sharing).
 * Window (1024B): ready@0, ack@64, numa@72, socket@76, smart votes 80..111,
 * then five [nranks]×int64 arrival arrays: AR stage@128, AR ①@256, AR ②@384,
 * AG stage@512, AG fan@640. Signal targets are the reverse partner relation
 * (rev_partners): under relaxed clamping "w waits me" is not symmetric with
 * "I wait w". */
constexpr size_t DB_ARR_AR_STAGE = 128;
constexpr size_t DB_ARR_AR_P1 = 256;
constexpr size_t DB_ARR_AR_P2 = 384;
constexpr size_t DB_ARR_AG_STAGE = 512;
constexpr size_t DB_ARR_AG_FAN = 640;

/* producer: publish seq into every waiter's local arrival slot. Self is
 * included on purpose: a local store, and it keeps the array complete for
 * waiters whose set contains the producer itself (group sets do). */
inline void db_signal(KuplComm *ctx, size_t arr_off, uint64_t seq, const int *waiters, size_t n)
{
    const int me = ctx->owner->rank;
    const size_t my_slot = sizeof(int64_t) * static_cast<size_t>(me);
    for (size_t i = 0; i < n; i++) {
        auto *slot = reinterpret_cast<volatile int64_t *>(
            reinterpret_cast<volatile char *>(ctx->peer_ready[waiters[i]]) + arr_off + my_slot);
        __atomic_store_n(slot, static_cast<int64_t>(seq), __ATOMIC_RELEASE);
    }
}

/* consumer: spin on OWN slots (local memory) until each producer published
 * seq. Self is skipped: my own phase is complete by construction when I am
 * the one waiting on it. */
inline void db_wait_local(KuplComm *ctx, size_t arr_off, const int *producers, size_t n, uint64_t seq)
{
    const int me = ctx->owner->rank;
    for (size_t i = 0; i < n; i++) {
        if (producers[i] == me)
            continue;
        auto *slot = reinterpret_cast<volatile int64_t *>(ctx->notify_base + arr_off +
                                                          sizeof(int64_t) * static_cast<size_t>(producers[i]));
        spin_wait(slot, seq);
    }
}

typedef void (*ParTaskFn)(kupl_nd_range_t *, void *, int, int);

static kupl_nd_range_t range_1d(int64_t lower, int64_t upper)
{
    kupl_nd_range_t nd;
    std::memset(&nd, 0, sizeof(nd));
    nd.dim = 1;
    nd.nd_range[0].lower = lower;
    nd.nd_range[0].upper = upper;
    nd.nd_range[0].step = 1;
    nd.nd_range[0].blocksize = 1;
    return nd;
}

static bool run_parallel(int64_t total, ParTaskFn fn, void *args, int rank)
{
    if (total <= 1 || g_config.parallel_threads <= 1)
        return false;
    static thread_local bool s_conc_set = false;
    if (!s_conc_set) {
        /* Concurrency must be ≤ executor count (or the DEFAULT sentinel). */
        const int cap = kupl_get_max_concurrency();
        kupl_set_kernel_concurrency(cap > 0 && g_config.parallel_threads > cap ? cap : g_config.parallel_threads);
        s_conc_set = true;
    }
    kupl_nd_range_t range = range_1d(0, total);
    kupl_parallel_for_desc_t desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT;
    desc.range = &range;
    desc.egroup = nullptr;
    desc.concurrency = KUPL_CONCURRENCY_DEFAULT;
    desc.policy = KUPL_LOOP_POLICY_STATIC;
    if (kupl_parallel_for(&desc, fn, args) == KUPL_OK)
        return true;
    static bool s_warned = false;
    if (!s_warned) {
        KuCCL_LOG_WARN(rank, "kupl_parallel_for unavailable, fallback to serial (warn once)");
        s_warned = true;
    }
    return false;
}

/* Serial equivalent of run_parallel: one full-range invocation. */
static void run_serial(int64_t total, ParTaskFn fn, void *args)
{
    kupl_nd_range_t nd = range_1d(0, total);
    fn(&nd, args, 0, 1);
}

/* ── Staging: grain-parallel local copy ───────────────────────*/

/* Staging is pure local bandwidth: one memcpy stream saturates a core well
 * below die bandwidth, so large chunks split into 8KB grains across the
 * executor. The threshold sits above the read-parallel floor — local copies
 * pay dispatch cost sooner than remote reads hide latency (breakeven ≈ 512K
 * at 4 threads). */
constexpr uint64_t STAGE_PAR_MIN_BYTES = 512 * 1024;
constexpr int64_t STAGE_GRAIN_BYTES = 8192;

struct StageCtx {
    char *dst = nullptr;
    const char *src = nullptr;
    size_t bytes = 0;
    volatile int failed = 0;
};

static void stage_task(kupl_nd_range_t *nd, void *args, int, int)
{
    auto *st = static_cast<StageCtx *>(args);
    const kupl_range_t &range = nd->nd_range[0];
    for (int64_t i = range.lower; i < range.upper && !st->failed; i += range.step) {
        const size_t off = static_cast<size_t>(i) * STAGE_GRAIN_BYTES;
        const size_t len = std::min<size_t>(STAGE_GRAIN_BYTES, st->bytes - off);
        if (kupl_memcpy(st->dst + off, st->src + off, len) != KUPL_OK)
            st->failed = 1;
    }
}

/* Returns false on copy failure (caller errors out). Executor refusal
 * degrades silently to the serial path — same policy as read parallelization. */
static bool stage_copy(char *dst, const char *src, size_t bytes, int rank)
{
    if (bytes >= STAGE_PAR_MIN_BYTES && g_config.parallel_threads > 1) {
        StageCtx st;
        st.dst = dst;
        st.src = src;
        st.bytes = bytes;
        const int64_t grains = static_cast<int64_t>((bytes + STAGE_GRAIN_BYTES - 1) / STAGE_GRAIN_BYTES);
        if (run_parallel(grains, stage_task, &st, rank))
            return !st.failed;
    }
    return kupl_memcpy(dst, src, bytes) == KUPL_OK;
}

/* ── Peer reads (flag + fence paths share this) ───────────────*/

struct PeerReadCtx {
    char *const *send_ptrs = nullptr;              /* peer send bases (cached) */
    volatile int64_t *const *peer_ready = nullptr; /* nullptr → fence path, no wait */
    uint64_t round = 0;
    size_t buf_off = 0; /* peer's round buffer offset (flag double buffer) */
    char *recvbuff = nullptr;
    size_t chunk = 0;
    int my_rank = 0;
    int failed_peer = -1;      /* parallel tasks write identical values — benign */
    bool serial_timing = true; /* g_timing is not thread-safe: no per-peer timing when parallel */
};

static void read_peer_chunk(PeerReadCtx *read_ctx, int peer)
{
    if (peer == read_ctx->my_rank)
        return;
    if (read_ctx->peer_ready != nullptr) {
        TimingScope tw(read_ctx->serial_timing && timing_enabled(), g_timing.ns[PHASE_FLAG_WAIT],
                       g_timing.cnt[PHASE_FLAG_WAIT]);
        spin_wait(read_ctx->peer_ready[peer], read_ctx->round);
    }
    TimingScope td(read_ctx->serial_timing, g_timing.ns[PHASE_DATA_MOVE], g_timing.cnt[PHASE_DATA_MOVE]);
    if (kupl_memcpy(read_ctx->recvbuff + static_cast<size_t>(peer) * read_ctx->chunk,
                    read_ctx->send_ptrs[peer] + read_ctx->buf_off, read_ctx->chunk) != KUPL_OK)
        read_ctx->failed_peer = peer;
}

static void read_peers_task(kupl_nd_range_t *nd, void *args, int, int)
{
    auto *read_ctx = static_cast<PeerReadCtx *>(args);
    const kupl_range_t &range = nd->nd_range[0];
    for (int64_t i = range.lower; i < range.upper && read_ctx->failed_peer < 0; i += range.step)
        read_peer_chunk(read_ctx, static_cast<int>(i));
}

/* Read every peer's chunk into user recvbuff [nranks][chunk]. Large chunks
 * fan out over the KUPL executor; single-threaded reads outrun thread
 * overhead below the parallel threshold. */
static bool read_all_peers(kucclComm *comm, KuplComm *ctx, void *recvbuff, size_t chunk, uint64_t round, bool use_flag,
                           size_t buf_off)
{
    PeerReadCtx read_ctx;
    read_ctx.send_ptrs = ctx->send_ptrs.data();
    read_ctx.peer_ready = use_flag ? ctx->peer_ready.data() : nullptr;
    read_ctx.round = round;
    read_ctx.buf_off = buf_off;
    read_ctx.recvbuff = static_cast<char *>(recvbuff);
    read_ctx.chunk = chunk;
    read_ctx.my_rank = comm->rank;

    const bool par = g_config.parallel_threads > 1 && chunk >= g_config.parallel_min_bytes && comm->nranks > 2;
    read_ctx.serial_timing = !par;
    bool ran = false;
    if (par) {
        TimingScope tp(timing_enabled(), g_timing.ns[PHASE_DATA_MOVE], g_timing.cnt[PHASE_DATA_MOVE]);
        ran = run_parallel(comm->nranks, read_peers_task, &read_ctx, comm->rank);
        if (!ran)
            read_ctx.failed_peer = -1; /* executor refused before running anything */
    }
    if (!ran) {
        read_ctx.serial_timing = true;
        run_serial(comm->nranks, read_peers_task, &read_ctx);
    }
    if (read_ctx.failed_peer >= 0) {
        detail::err_record("kupl allgather: read peer=%d FAILED (kupl_memcpy)", read_ctx.failed_peer);
        return false;
    }
    return true;
}

/* ── doorbell: zero global fence (opt-in via KUCCL_KUPL_AG_SYNC) ──*/

static kucclResult_t allgather_doorbell(kucclComm *comm, KuplComm *ctx, const CollParams *params, size_t chunk)
{
    const int my_rank = comm->rank;
    const uint64_t round = ctx->flag_round + 1;

    /* ① drain: peers finished reading the round we are about to overwrite
     *    (double-buffer parity protects only one round of lag) */
    drain_peer_acks(comm, ctx);

    /* ② capacity: double buffer */
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_SCRATCH], g_timing.cnt[PHASE_SCRATCH]);
        if (ctx->send_cap < chunk * 2 && ensure_send(ctx, chunk * 2) != KuCCL_SUCCESS) {
            abort_flag_round(ctx, round);
            return KuCCL_FAIL;
        }
    }
    const size_t half_bytes = ctx->send_cap / 2;
    const size_t buf_off = static_cast<size_t>(round & 1) * half_bytes;
    char *round_buf = ctx->send_base + buf_off;

    /* ③ stage twice: round buffer (peers read here) + own slot in user buffer */
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_STAGE_OUT], g_timing.cnt[PHASE_STAGE_OUT]);
        if (!stage_copy(round_buf, static_cast<const char *>(params->sendbuff), chunk, my_rank) ||
            !stage_copy(static_cast<char *>(params->recvbuff) + static_cast<size_t>(my_rank) * chunk,
                        static_cast<const char *>(params->sendbuff), chunk, my_rank)) {
            detail::err_record("kupl allgather(flag): stage memcpy(%zuB) FAILED", chunk);
            abort_flag_round(ctx, round);
            return KuCCL_FAIL;
        }
    }

    /* ④ doorbell (RELEASE: data lands before the bell rings) */
    __atomic_store_n(ctx->ready_flag, static_cast<int64_t>(round), __ATOMIC_RELEASE);

    /* ⑤ per-peer doorbell → direct read into user buffer */
    if (!read_all_peers(comm, ctx, params->recvbuff, chunk, round, true, buf_off)) {
        abort_flag_round(ctx, round);
        return KuCCL_FAIL;
    }

    /* ⑥ ack this round (next round's drain anchor) */
    __atomic_store_n(ctx->ack_flag, static_cast<int64_t>(round), __ATOMIC_RELEASE);
    ctx->flag_round = round;
    KuCCL_LOG_TRACE(comm->rank, "kupl allgather(flag): done (round=%llu, chunk=%zuB)",
                    static_cast<unsigned long long>(round), chunk);
    return KuCCL_SUCCESS;
}

/* ── fence: full fence ×2 + single buffer (conservative) ──*/

static kucclResult_t allgather_fence(kucclComm *comm, KuplComm *ctx, const CollParams *params, size_t chunk)
{
    const int my_rank = comm->rank;

    /* flag rounds carry no trailing fence: staging at offset 0 overlaps the
     * flag path's even-round buffer — the ack drain is the safety net */
    drain_peer_acks(comm, ctx);
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_SCRATCH], g_timing.cnt[PHASE_SCRATCH]);
        if (ensure_send(ctx, chunk) != KuCCL_SUCCESS)
            return KuCCL_FAIL;
    }
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_STAGE_OUT], g_timing.cnt[PHASE_STAGE_OUT]);
        if (!stage_copy(ctx->send_base, static_cast<const char *>(params->sendbuff), chunk, my_rank) ||
            !stage_copy(static_cast<char *>(params->recvbuff) + static_cast<size_t>(my_rank) * chunk,
                        static_cast<const char *>(params->sendbuff), chunk, my_rank)) {
            detail::err_record("kupl allgather(fence): stage memcpy(%zuB) FAILED", chunk);
            return KuCCL_FAIL;
        }
    }

    timed_fence(ctx->send_win, comm->rank); /* head: all stagings landed */
    if (!read_all_peers(comm, ctx, params->recvbuff, chunk, 0, false, 0))
        return KuCCL_FAIL;
    timed_fence(ctx->send_win, comm->rank); /* tail: slow readers left before
                                              the next round overwrites */
    return KuCCL_SUCCESS;
}

/* ── die_pair: NUMA-aware pair exchange (large-message default) ──*/

struct FanoutCtx {
    char *const *send_ptrs = nullptr;
    char *const *recv_ptrs = nullptr;
    const int *src_ranks = nullptr; /* K sources: [0]=own rank, [1..]=partners */
    const int *dst_ranks = nullptr; /* L local peers (ascending, incl. self) */
    int num_sources = 0;
    int num_dsts = 0;
    int num_tiles = 0;
    size_t chunk = 0;
    volatile int failed = 0;
};

/* One (source, tile) task body: read the source tile once, write it into
 * every local peer's landing slot for that source. Tiling keeps cross-node
 * reads un-amplified while giving the executor enough parallelism. */
static void fanout_one(FanoutCtx *fan, int64_t i)
{
    const size_t tile = (fan->chunk + static_cast<size_t>(fan->num_tiles) - 1) / static_cast<size_t>(fan->num_tiles);
    const int src_idx = static_cast<int>(i) / fan->num_tiles;
    const size_t tile_off = static_cast<size_t>(i % fan->num_tiles) * tile;
    const size_t tile_len = std::min(tile, fan->chunk - tile_off);
    const char *src_ptr = fan->send_ptrs[fan->src_ranks[src_idx]] + tile_off;
    for (int dst_idx = 0; dst_idx < fan->num_dsts; dst_idx++) {
        char *dst_ptr = fan->recv_ptrs[fan->dst_ranks[dst_idx]] +
                        static_cast<size_t>(fan->src_ranks[src_idx]) * fan->chunk + tile_off;
        if (kupl_memcpy(dst_ptr, src_ptr, tile_len) != KUPL_OK) {
            fan->failed = 1;
            return;
        }
    }
}

/* ── hybrid pair: home-group linear reads + cross-group fanout ──
 *
 * Pure pair_exchange shuttles every chunk through the landing zone —
 * including the home group's, which linear already reads directly. The hybrid
 * keeps the landing zone for cross-group chunks only:
 *   A) home-group members' chunks are read straight into recvbuff;
 *   B) each rank reads its partners' staged chunk once and fans it into the
 *      home group's landing slots ([partner_rank] slots of [nranks][chunk]);
 *   C) only the partner slots are copied back, parallel per slot.
 * Cross-boundary traffic is unchanged (one crossing per chunk). */

struct GroupReadCtx {
    char *const *send_ptrs = nullptr;
    const char *self_src = nullptr; /* my own sendbuff (self slot) */
    char *recvbuff = nullptr;
    const int *peers = nullptr; /* home-group ranks (incl. self) */
    int num_peers = 0;
    int my_rank = 0;
    size_t chunk = 0;
    volatile int failed = 0;
};

static void group_read_one(GroupReadCtx *g, int64_t i)
{
    const int p = g->peers[static_cast<size_t>(i)];
    const char *src = (p == g->my_rank) ? g->self_src : g->send_ptrs[static_cast<size_t>(p)];
    if (kupl_memcpy(g->recvbuff + static_cast<size_t>(p) * g->chunk, src, g->chunk) != KUPL_OK)
        g->failed = 1;
}

/* Merged A+B dispatch: home-group reads and cross-group fanout have no data
 * dependency (different destination buffers), so one scheduling round runs
 * both — the local reads overlap the cross-boundary read latency. */
struct HybridDispatch {
    GroupReadCtx *gr = nullptr;
    FanoutCtx *fan = nullptr;
    int64_t a_tasks = 0;
    volatile int failed = 0;
};

static void hybrid_task(kupl_nd_range_t *nd, void *args, int, int)
{
    auto *h = static_cast<HybridDispatch *>(args);
    const kupl_range_t &range = nd->nd_range[0];
    for (int64_t i = range.lower; i < range.upper && !h->failed; i += range.step) {
        if (i < h->a_tasks) {
            group_read_one(h->gr, i);
            if (h->gr->failed)
                h->failed = 1;
        } else {
            fanout_one(h->fan, i - h->a_tasks);
            if (h->fan->failed)
                h->failed = 1;
        }
    }
}

struct CopyBackCtx {
    const char *recv_base = nullptr;
    char *recvbuff = nullptr;
    const int *slots = nullptr; /* partner ranks whose landing slots hold data */
    int num_slots = 0;
    size_t chunk = 0;
    volatile int failed = 0;
};

static void copy_back_task(kupl_nd_range_t *nd, void *args, int, int)
{
    auto *c = static_cast<CopyBackCtx *>(args);
    const kupl_range_t &range = nd->nd_range[0];
    for (int64_t i = range.lower; i < range.upper && !c->failed; i += range.step) {
        const int p = c->slots[static_cast<size_t>(i)];
        if (kupl_memcpy(c->recvbuff + static_cast<size_t>(p) * c->chunk,
                        c->recv_base + static_cast<size_t>(p) * c->chunk, c->chunk) != KUPL_OK)
            c->failed = 1;
    }
}

static kucclResult_t allgather_pair_hybrid(kucclComm *comm, KuplComm *ctx, const CollParams *params, size_t chunk,
                                           const std::vector<int> &local_peers, const std::vector<int> &partners,
                                           const std::vector<int> &rev_partners, const std::vector<int> &remote_ranks,
                                           const char *tag)
{
    const size_t total = chunk * static_cast<size_t>(comm->nranks);

    drain_peer_acks(comm, ctx);
    const uint64_t seq = ++ctx->db_seq; /* doorbell round; hier paths bump it in lockstep */
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_SCRATCH], g_timing.cnt[PHASE_SCRATCH]);
        if (ensure_send(ctx, chunk) != KuCCL_SUCCESS)
            return KuCCL_FAIL; /* partners AND home group read my chunk from send_win */
        if (ensure_recv(ctx, total) != KuCCL_SUCCESS)
            return KuCCL_FAIL; /* landing zone [nranks][chunk]; only partner slots used */
    }
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_STAGE_OUT], g_timing.cnt[PHASE_STAGE_OUT]);
        if (!stage_copy(ctx->send_base, static_cast<const char *>(params->sendbuff), chunk, comm->rank)) {
            detail::err_record("kupl allgather(%s): stage memcpy(%zuB) FAILED", tag, chunk);
            db_signal(ctx, DB_ARR_AG_STAGE, seq, local_peers.data(), local_peers.size());
            db_signal(ctx, DB_ARR_AG_STAGE, seq, rev_partners.data(), rev_partners.size());
            db_signal(ctx, DB_ARR_AG_FAN, seq, local_peers.data(), local_peers.size());
            return KuCCL_FAIL;
        }
    }
    /* stagings visible: doorbell — phase A reads home mates' staging, phase B
     * fetches partners' staging. Signal both sets (local stores for mates,
     * remote stores for reverse partners), then spin locally on the union. */
    db_signal(ctx, DB_ARR_AG_STAGE, seq, local_peers.data(), local_peers.size());
    db_signal(ctx, DB_ARR_AG_STAGE, seq, rev_partners.data(), rev_partners.size());
    db_wait_local(ctx, DB_ARR_AG_STAGE, local_peers.data(), local_peers.size(), seq);
    db_wait_local(ctx, DB_ARR_AG_STAGE, partners.data(), partners.size(), seq);

    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_DATA_MOVE], g_timing.cnt[PHASE_DATA_MOVE]);

        /* A) home-group linear reads: straight into recvbuff */
        GroupReadCtx gr;
        gr.send_ptrs = ctx->send_ptrs.data();
        gr.self_src = static_cast<const char *>(params->sendbuff);
        gr.recvbuff = static_cast<char *>(params->recvbuff);
        gr.peers = local_peers.data();
        gr.num_peers = static_cast<int>(local_peers.size());
        gr.my_rank = comm->rank;
        gr.chunk = chunk;

        /* B) cross-group fanout: read each partner's staged chunk once, fan
         * it into the home group's landing slots. 4KB tiles keep cross-group
         * reads un-amplified (total bytes read = chunk) while giving the
         * executor enough parallelism (task count capped at 32). */
        FanoutCtx fan;
        fan.send_ptrs = ctx->send_ptrs.data();
        fan.recv_ptrs = ctx->recv_ptrs.data();
        fan.src_ranks = partners.data();
        fan.dst_ranks = local_peers.data();
        fan.num_sources = static_cast<int>(partners.size());
        fan.num_dsts = static_cast<int>(local_peers.size());
        fan.num_tiles = static_cast<int>(std::min<uint64_t>(std::max<uint64_t>(chunk / 4096, 1), 32));
        fan.chunk = chunk;

        /* A+B in one scheduling round: home reads overlap cross-group reads */
        HybridDispatch h;
        h.gr = &gr;
        h.fan = &fan;
        h.a_tasks = static_cast<int64_t>(gr.num_peers);
        const int64_t tasks = h.a_tasks + static_cast<int64_t>(fan.num_sources) * fan.num_tiles;
        if (!run_parallel(tasks, hybrid_task, &h, comm->rank))
            run_serial(tasks, hybrid_task, &h);
        if (h.failed) {
            detail::err_record("kupl allgather(%s): data move FAILED", tag);
            db_signal(ctx, DB_ARR_AG_FAN, seq, local_peers.data(), local_peers.size());
            return KuCCL_FAIL;
        }
    }
    /* fanout (into home mates' landing slots) visible: doorbell instead of a
     * full barrier. Landing slot [p] of mate m is written by the home member
     * whose partner-in-group-j is p — so copy-back waits for every home
     * mate's flag (self writes are self-visible, skipped). */
    db_signal(ctx, DB_ARR_AG_FAN, seq, local_peers.data(), local_peers.size());
    db_wait_local(ctx, DB_ARR_AG_FAN, local_peers.data(), local_peers.size(), seq);

    {
        /* C) parallel copy-back of partner slots only (home slots are
         * already in place from stage A) */
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_COPY_BACK], g_timing.cnt[PHASE_COPY_BACK]);
        CopyBackCtx cb;
        cb.recv_base = ctx->recv_base;
        cb.recvbuff = static_cast<char *>(params->recvbuff);
        cb.slots = remote_ranks.data();
        cb.num_slots = static_cast<int>(remote_ranks.size());
        cb.chunk = chunk;
        if (!run_parallel(static_cast<int64_t>(cb.num_slots), copy_back_task, &cb, comm->rank))
            run_serial(static_cast<int64_t>(cb.num_slots), copy_back_task, &cb);
        if (cb.failed) {
            detail::err_record("kupl allgather(%s): copy-back memcpy FAILED", tag);
            return KuCCL_FAIL;
        }
    }
    timed_fence(ctx->recv_win, comm->rank); /* recycle barrier: copy-backs done
                                              before anyone rewrites the zone */
    KuCCL_LOG_TRACE(comm->rank, "kupl allgather(%s hybrid): done (home=%d partners=%d remote=%d chunk=%zuB)", tag,
                    static_cast<int>(local_peers.size()), static_cast<int>(partners.size()),
                    static_cast<int>(remote_ranks.size()), chunk);
    return KuCCL_SUCCESS;
}

static kucclResult_t allgather_die_pair(kucclComm *comm, KuplComm *ctx, const CollParams *params, size_t chunk)
{
    /* Die-granular pairing scales to cross-socket layouts too: partners span
     * every remote die (both sockets), so each chunk crosses each boundary
     * exactly once while the fanout stays inside the home die. */
    return allgather_pair_hybrid(comm, ctx, params, chunk, ctx->topo.local_ranks, ctx->topo.partners,
                                 ctx->topo.rev_partners, ctx->topo.remote_ranks, "die_pair");
}

/* socket_pair — two-level hierarchy: socket-granular pair exchange. The cross-socket hop
 * (the narrow, shared interconnect) carries each chunk exactly once per remote
 * socket; spreading within my socket rides the cheaper die hops. */
static kucclResult_t allgather_socket_pair(kucclComm *comm, KuplComm *ctx, const CollParams *params, size_t chunk)
{
    return allgather_pair_hybrid(comm, ctx, params, chunk, ctx->topo.socket_local_ranks, ctx->topo.socket_partners,
                                 ctx->topo.socket_rev_partners, ctx->topo.socket_remote_ranks, "socket_pair");
}

/* ── smart routing: lazy learning on real calls ── */

static int smart_bucket(uint64_t chunk)
{
    int b = 0;
    while ((chunk >>= 1) != 0)
        b++;
    return b >= SmartRoute::NUM_BUCKETS ? SmartRoute::NUM_BUCKETS - 1 : b;
}

/* Candidate set derives only from collective-consistent topology flags. */
static void smart_build_cands(KuplComm *ctx)
{
    SmartRoute &s = ctx->smart;
    if (s.cand_built)
        return;
    s.cand_map[0] = {static_cast<uint8_t>(ALLGATHER_LINEAR), false}; /* linear + fence */
    s.cand_map[1] = {static_cast<uint8_t>(ALLGATHER_LINEAR), true};  /* linear + doorbell */
    s.cand_total = 2;
    if (ctx->topo.usable)
        s.cand_map[s.cand_total++] = {static_cast<uint8_t>(ALLGATHER_DIE_PAIR), false};
    if (ctx->topo.socket_usable)
        s.cand_map[s.cand_total++] = {static_cast<uint8_t>(ALLGATHER_SOCKET_PAIR), false};
    s.cand_built = true;
}

static int smart_cand_idx(const SmartRoute &s, AgChoice used)
{
    for (int i = 0; i < s.cand_total; i++)
        if (s.cand_map[i].algo == used.algo && s.cand_map[i].doorbell == used.doorbell)
            return i;
    return -1;
}

static AgChoice smart_next(kucclComm *comm, KuplComm *ctx, uint64_t chunk)
{
    (void)comm;
    SmartRoute &s = ctx->smart;
    smart_build_cands(ctx);
    const int b = smart_bucket(chunk);
    if (s.phase[b] == 0) {
        s.phase[b] = 1;
        /* first call in the bucket: the auto router's choice, seeded with
         * the configured sync primitive */
        s.choice[b] = {static_cast<uint8_t>(g_config.route(ctx->topo.usable, ctx->topo.socket_usable)),
                       g_config.sync_doorbell};
        return s.choice[b];
    }
    if (s.phase[b] == 2)
        return s.choice[b];
    for (int i = 0; i < s.cand_total; i++)
        if (s.samples[b][i] < static_cast<uint16_t>(g_config.smart_reps))
            return s.cand_map[i];
    return s.choice[b]; /* unreachable: observe flips the bucket to learned */
}

static void smart_observe(kucclComm *comm, KuplComm *ctx, uint64_t chunk, AgChoice used, uint64_t elapsed_ns)
{
    SmartRoute &s = ctx->smart;
    const int b = smart_bucket(chunk);
    if (s.phase[b] != 1)
        return;
    const int idx = smart_cand_idx(s, used);
    if (idx < 0)
        return;
    s.sum_ns[b][idx] += elapsed_ns;
    s.samples[b][idx]++;
    for (int i = 0; i < s.cand_total; i++)
        if (s.samples[b][i] < static_cast<uint16_t>(g_config.smart_reps))
            return;

    auto *mine = reinterpret_cast<volatile uint64_t *>(ctx->notify_base + 80);
    for (int i = 0; i < s.cand_total; i++)
        mine[i] = s.sum_ns[b][i] / s.samples[b][i];
    kupl_shm_fence(ctx->notify_win);
    uint64_t worst[SmartRoute::MAX_CAND] = {};
    for (int peer = 0; peer < comm->nranks; peer++) {
        /* keep the volatile qualifier through the cast chain — casting it
         * away is a hard error on clang (and a real hazard: peer scratch is
         * written by another process) */
        const auto *peers =
            reinterpret_cast<volatile uint64_t *>(reinterpret_cast<volatile char *>(ctx->peer_ready[peer]) + 80);
        for (int i = 0; i < s.cand_total; i++) {
            const uint64_t sample = peers[i]; /* one volatile read; std::max takes refs and
                                                 cannot deduce volatile vs plain _Tp */
            worst[i] = std::max(worst[i], sample);
        }
    }
    kupl_shm_fence(ctx->notify_win);
    int best = 0;
    for (int i = 1; i < s.cand_total; i++)
        if (worst[i] < worst[best])
            best = i;
    s.choice[b] = s.cand_map[best];
    s.phase[b] = 2;
    KuCCL_LOG_INFO(comm->rank, "kupl smart-route: ~%lluB → %s (avg %.1f us, worst-rank vote)",
                   static_cast<unsigned long long>(1ULL << smart_bucket(chunk)),
                   ag_name(s.choice[b].algo, s.choice[b].doorbell), static_cast<double>(worst[best]) / 1000.0);
}

/* ── AllGather entry: validate → route → run ──*/

static kucclResult_t impl_allgather(kucclComm *comm, KuplComm *ctx, const CollParams *params, size_t dt_size,
                                    KuplReq *priv, kupl_shm_request_h *native_req)
{
    TimingScope t_total(timing_enabled(), g_timing.ns_op[OP_ALLGATHER], g_timing.cnt_op[OP_ALLGATHER]);
    const uint64_t t0 = (timing_enabled() || g_config.smart_route) ? KuplTiming::now() : 0;
    const size_t chunk = params->count * dt_size;
    const size_t total = chunk * static_cast<size_t>(comm->nranks);
    if (chunk == 0 || total > static_cast<size_t>(INT32_MAX)) {
        detail::err_record("kupl allgather: invalid chunk=%zu total=%zu", chunk, total);
        return KuCCL_FAIL;
    }

    /* smart route: static choice first, repeats explore, worst-rank vote settles */
    AgChoice c;
    if (g_config.smart_route) {
        c = smart_next(comm, ctx, chunk);
    } else {
        c = {static_cast<uint8_t>(g_config.route(ctx->topo.usable, ctx->topo.socket_usable)), g_config.sync_doorbell};
    }
    KuCCL_LOG_TRACE(comm->rank, "[AG-route] chunk=%zuB ws=%d → %s", chunk, comm->nranks, ag_name(c.algo, c.doorbell));

    kucclResult_t ret;
    switch (static_cast<AllgatherAlgo>(c.algo)) {
        case ALLGATHER_DIE_PAIR:
            ret = allgather_die_pair(comm, ctx, params, chunk);
            break;
        case ALLGATHER_SOCKET_PAIR:
            ret = allgather_socket_pair(comm, ctx, params, chunk);
            break;
        default:
            ret = c.doorbell ? allgather_doorbell(comm, ctx, params, chunk) : allgather_fence(comm, ctx, params, chunk);
            break;
    }
    if (ret != KuCCL_SUCCESS) {
        /* Observe failures too: a penalty sample keeps per-bucket sample
         * counts identical on every rank (counts drive the exploration
         * schedule — divergence = deadlock). */
        if (g_config.smart_route)
            smart_observe(comm, ctx, chunk, c, 1ULL << 40);
        return ret;
    }

    /* result already in the user buffer — nothing to copy back */
    priv->user_dst = nullptr;
    priv->win_src = nullptr;
    priv->copy_bytes = 0;
    *native_req = nullptr;
    if (g_config.smart_route) {
        const uint64_t elapsed = KuplTiming::now() - t0;
        if (timing_enabled())
            record_allgather_latency(chunk, elapsed);
        smart_observe(comm, ctx, chunk, c, elapsed);
    } else if (timing_enabled()) {
        record_allgather_latency(chunk, KuplTiming::now() - t0);
    }
    return KuCCL_SUCCESS;
}

/* SUM → KUPL native fast path: stage send → allreduce (synchronous start)
 * → result in recv window → req_test copies back. */
static kucclResult_t allreduce_sum_native(kucclComm *comm, KuplComm *ctx, const CollParams *params, size_t bytes,
                                          KuplReq *priv, kupl_shm_request_h *native_req)
{
    drain_peer_acks(comm, ctx);
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_SCRATCH], g_timing.cnt[PHASE_SCRATCH]);
        if (ensure_send(ctx, bytes) != KuCCL_SUCCESS || ensure_recv(ctx, bytes) != KuCCL_SUCCESS) {
            detail::err_record("kupl allreduce(SUM): staging FAILED (bytes=%zu)", bytes);
            return KuCCL_FAIL;
        }
    }
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_STAGE_OUT], g_timing.cnt[PHASE_STAGE_OUT]);
        if (kupl_memcpy(ctx->send_base, params->sendbuff, bytes) != KUPL_OK) {
            detail::err_record("kupl allreduce(SUM): stage memcpy(%zuB) FAILED", bytes);
            return KuCCL_FAIL;
        }
    }
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_NATIVE], g_timing.cnt[PHASE_NATIVE]);
        if (kupl_shm_allreduce_init(ctx->send_base, ctx->recv_base, static_cast<int>(params->count),
                                    map_datatype(params->datatype), KUPL_SHM_REDUCE_OP_SUM, ctx->comm,
                                    native_req) != 0) {
            detail::err_record("kupl allreduce(SUM): init FAILED count=%zu dt=%d", params->count,
                               static_cast<int>(params->datatype));
            return KuCCL_FAIL;
        }
        if (kupl_shm_request_start(*native_req) != 0) {
            detail::err_record("kupl allreduce(SUM): start FAILED");
            kupl_shm_request_free(*native_req);
            *native_req = nullptr;
            return KuCCL_FAIL;
        }
    }
    priv->user_dst = params->recvbuff;
    priv->win_src = ctx->recv_base;
    priv->copy_bytes = bytes;
    KuCCL_LOG_TRACE(comm->rank, "kupl allreduce SUM: done (count=%zu)", params->count);
    return KuCCL_SUCCESS;
}

static kucclResult_t allreduce_minmax_recursive(kucclComm *comm, KuplComm *ctx, const CollParams *params,
                                                size_t dt_size, size_t bytes, KuplReq *priv)
{
    const int count = static_cast<int>(params->count);
    int nsteps = 0;
    while ((1 << nsteps) < comm->nranks)
        nsteps++;

    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_STAGE_OUT], g_timing.cnt[PHASE_STAGE_OUT]);
        if (kupl_memcpy(ctx->recv_base, params->sendbuff, bytes) != KUPL_OK) {
            detail::err_record("kupl allreduce(RH): init memcpy(%zuB) FAILED", bytes);
            return KuCCL_FAIL;
        }
    }

    /* Act I — halving: fold partners' copies of my range into mine */
    for (int k = 0; k < nsteps; k++) {
        const int partner = comm->rank ^ (1 << k);
        int lo, hi;
        calc_range(comm->rank, count, k, &lo, &hi);
        timed_fence(ctx->recv_win, comm->rank);
        if (lo > hi)
            continue; /* count < nranks: my share of this step is empty */
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_DATA_MOVE], g_timing.cnt[PHASE_DATA_MOVE]);
        dispatch_minmax(params->datatype, params->redop, ctx->recv_base + static_cast<size_t>(lo) * dt_size,
                        ctx->recv_ptrs[partner] + static_cast<size_t>(lo) * dt_size, hi - lo + 1);
    }
    timed_fence(ctx->recv_win, comm->rank); /* intermission: Act I complete */

    /* Act II — doubling: copy partners' authoritative ranges back */
    for (int k = nsteps - 1; k >= 0; k--) {
        const int partner = comm->rank ^ (1 << k);
        int lo, hi;
        calc_range(partner, count, k, &lo, &hi);
        timed_fence(ctx->recv_win, comm->rank);
        if (lo > hi)
            continue;
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_DATA_MOVE], g_timing.cnt[PHASE_DATA_MOVE]);
        const size_t len = static_cast<size_t>(hi - lo + 1) * dt_size;
        if (kupl_memcpy(ctx->recv_base + static_cast<size_t>(lo) * dt_size,
                        ctx->recv_ptrs[partner] + static_cast<size_t>(lo) * dt_size, len) != KUPL_OK) {
            detail::err_record("kupl allreduce(RD): memcpy(%zuB) FAILED", len);
            return KuCCL_FAIL;
        }
    }
    timed_fence(ctx->recv_win, comm->rank); /* epilogue */

    priv->user_dst = params->recvbuff;
    priv->win_src = ctx->recv_base;
    priv->copy_bytes = bytes;
    return KuCCL_SUCCESS;
}

/* MIN/MAX, direct: copy peer 0's raw data, then fold every peer element-wise.
 * O(N) passes, no protocol subtleties — the small/skew-group fallback. */
static kucclResult_t allreduce_minmax_direct(kucclComm *comm, KuplComm *ctx, const CollParams *params, size_t bytes,
                                             KuplReq *priv)
{
    timed_fence(ctx->send_win, comm->rank);
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_DATA_MOVE], g_timing.cnt[PHASE_DATA_MOVE]);
        for (int peer = 0; peer < comm->nranks; peer++) {
            const char *peer_send = ctx->send_ptrs[peer];
            if (peer == 0) {
                if (kupl_memcpy(ctx->recv_base, peer_send, bytes) != KUPL_OK) {
                    detail::err_record("kupl allreduce(MIN/MAX): memcpy(peer=0, %zuB) FAILED", bytes);
                    return KuCCL_FAIL;
                }
            } else {
                dispatch_minmax(params->datatype, params->redop, ctx->recv_base, peer_send,
                                static_cast<int>(params->count));
            }
        }
    }
    timed_fence(ctx->send_win, comm->rank);

    priv->user_dst = params->recvbuff;
    priv->win_src = ctx->recv_base;
    priv->copy_bytes = bytes;
    return KuCCL_SUCCESS;
}

/* ── layered AllReduce: intra-group reduce-scatter → o/e sub-block exchange →
 * owner fanout. One engine, two granularities (mirrors the allgather pair
 * family). ──
 *
 * Write/read discipline — folds are index-disjoint, never in-place on data
 * other ranks still read:
 *   stage  sendbuf → own send segment (group peers read the raw slice in ①)
 *   ①      the member with group index i < M owns slice i (block split of
 *          count into M blocks, remainder spread over the leading slices)
 *          and folds every group peer's staged data for that slice into its
 *          own recv segment → group partial for the whole slice
 *   ②      the slice is further split into K sub-blocks (K = group count),
 *          sub-block j owned by group j's slice owner. Each owner folds
 *          every partner's partial at its own sub-block into its own
 *          sub-block in place — safe because every owner reads each remote
 *          segment only at its own sub-block index: what I write, nobody
 *          reads; what I read, nobody writes
 *   ③      the owner fetches the K−1 foreign sub-block globals from the
 *          partner owners, completing its slice as a full global, then fans
 *          the slice out to every group member (one writer per slice per
 *          group); members idle until the recycle fence
 *   req_test copies the segment back to the user buffer
 *
 * Slicing: M = smallest group size, so for i < M the
 * partner groups[j][i] exists in every group — the ② exchange stays
 * index-aligned by construction; members past M own no slice (they still
 * pull results in ③).
 *
 * Traffic per slice owner: 2(K−1)/K × slice cross reads. Every element's
 * global is computed exactly once (by its group's owner). Recv segment
 * stays 1×.
 *
 * Sync: fence after staging / ① / ② + a trailing recycle fence — 4 fences
 * per call. SUM for float/double folds in a different order than KUPL
 * native — bit-exact equality is not guaranteed (README documents this).
 *
 * Parallelism: ① and ② fold strip-parallel (dst-disjoint strips over the
 * executor; serial below the parallel floor); ③'s fetch stays serial, the
 * member fanout is executor-parallel. */

/* Block slice for group index idx of m blocks: elements [start, start+len).
 * idx outside [0, m) yields an empty slice — callers gate ownership with
 * idx < m; the clamp makes an unchecked idx fail safe instead of producing
 * out-of-range offsets. */
static void ar_hier_slice(size_t count, int idx, int m, size_t *start, size_t *len)
{
    if (m <= 0 || idx < 0 || idx >= m) {
        *start = 0;
        *len = 0;
        return;
    }
    const size_t div = count / static_cast<size_t>(m);
    const size_t rem = count % static_cast<size_t>(m);
    const size_t i = static_cast<size_t>(idx);
    *start = i * div + (i < rem ? i : rem);
    *len = div + (i < rem ? 1 : 0);
}

/* Slice count for the layered AllReduce (see block comment). */
static int ar_min_group_size(const std::vector<std::vector<int>> &all_groups)
{
    size_t m = all_groups[0].size();
    for (const auto &g : all_groups)
        m = std::min(m, g.size());
    return static_cast<int>(m);
}

/* ① / ② fold task: one STRIP of the destination range — seed from the self
 * source (①: the user sendbuff; ②: nullptr = fold into the ① partial in
 * place), then combine every peer's strip. Strips are dst-disjoint →
 * executor-safe; the strip split reuses ar_hier_slice's remainder spread. */
struct ArFoldCtx {
    kucclDataType_t dt;
    kucclRedOp_t op;
    size_t dt_size = 0;
    char *dst = nullptr;             /* my recv segment base */
    const char *self_src = nullptr;  /* ① seed source (user buffer base); ②: nullptr */
    char *const *seg_ptrs = nullptr; /* ①: peer send bases; ②: partner recv bases */
    const int *peers = nullptr;      /* ranks whose segments to fold */
    int num_peers = 0;
    int my_rank = 0;      /* ①: group includes self — skipped in-task */
    size_t range_off = 0; /* byte offset of the folded range inside every segment */
    size_t count = 0;     /* element count of the folded range (strips split this) */
    int strips = 0;
    volatile int failed = 0;
};

static void ar_fold_task(kupl_nd_range_t *nd, void *args, int, int)
{
    auto *f = static_cast<ArFoldCtx *>(args);
    const kupl_range_t &range = nd->nd_range[0];
    for (int64_t t = range.lower; t < range.upper && !f->failed; t += range.step) {
        size_t s0 = 0, sl = 0;
        ar_hier_slice(f->count, static_cast<int>(t), f->strips, &s0, &sl);
        if (sl == 0)
            continue;
        const size_t off = f->range_off + s0 * f->dt_size;
        char *dst = f->dst + off;
        if (f->self_src != nullptr && kupl_memcpy(dst, f->self_src + off, sl * f->dt_size) != KUPL_OK) {
            f->failed = 1;
            return;
        }
        for (int i = 0; i < f->num_peers; i++) {
            const int peer = f->peers[i];
            if (peer == f->my_rank)
                continue;
            dispatch_combine(f->dt, f->op, dst, f->seg_ptrs[peer] + off, static_cast<int>(sl));
        }
    }
}

/* Strip-parallel fold driver: above the parallel floor the strips fan out
 * over the executor; below it one serial invocation covers everything
 * (thread overhead dominates small folds). Combine itself cannot fail —
 * only the ① seed memcpy can. */
static bool ar_fold_run(ArFoldCtx &f, int rank)
{
    const bool par = g_config.parallel_threads > 1 && f.count * f.dt_size >= g_config.parallel_min_bytes;
    if (par && run_parallel(f.strips, ar_fold_task, &f, rank))
        return f.failed == 0;
    run_serial(f.strips, ar_fold_task, &f);
    return f.failed == 0;
}

/* ③ fanout task: push my completed global slice into one group member's
 * segment (disjoint destinations per task → executor-safe). */
struct ArPushCtx {
    char *const *recv_ptrs = nullptr; /* peer recv segment bases */
    const int *group = nullptr;       /* group member ranks (ascending, incl. self) */
    int my_rank = 0;
    const char *src = nullptr; /* my completed slice (global) */
    size_t dst_off = 0;        /* slice offset in every segment */
    size_t bytes = 0;          /* slice bytes */
    volatile int failed = 0;
};

static void ar_push_task(kupl_nd_range_t *nd, void *args, int, int)
{
    auto *p = static_cast<ArPushCtx *>(args);
    const kupl_range_t &range = nd->nd_range[0];
    for (int64_t i = range.lower; i < range.upper && !p->failed; i += range.step) {
        const int peer = p->group[i];
        if (peer == p->my_rank)
            continue;
        if (kupl_memcpy(p->recv_ptrs[peer] + p->dst_off, p->src, p->bytes) != KUPL_OK) {
            p->failed = 1;
            return;
        }
    }
}

static kucclResult_t allreduce_layered(kucclComm *comm, KuplComm *ctx, const CollParams *params, size_t dt_size,
                                       size_t bytes, KuplReq *priv, const std::vector<int> &group,
                                       const std::vector<int> &partners, const std::vector<int> &rev_partners,
                                       int my_group, const std::vector<std::vector<int>> &all_groups, const char *tag)
{
    const int members = static_cast<int>(group.size());
    const int slices = ar_min_group_size(all_groups);
    const int ngroups = static_cast<int>(all_groups.size());
    int my_idx = 0;
    for (int i = 0; i < members; i++) {
        if (group[static_cast<size_t>(i)] == comm->rank) {
            my_idx = i;
            break;
        }
    }

    drain_peer_acks(comm, ctx);
    const uint64_t seq = ++ctx->db_seq; /* doorbell round; hier paths bump it in lockstep
                                           (unconditionally: a rank that fails later still
                                           bumped, so the seq can never desync) */
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_SCRATCH], g_timing.cnt[PHASE_SCRATCH]);
        if (ensure_send(ctx, bytes) != KuCCL_SUCCESS || ensure_recv(ctx, bytes) != KuCCL_SUCCESS)
            return KuCCL_FAIL;
    }
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_STAGE_OUT], g_timing.cnt[PHASE_STAGE_OUT]);
        if (kupl_memcpy(ctx->send_base, params->sendbuff, bytes) != KUPL_OK) {
            detail::err_record("kupl allreduce(%s): stage memcpy(%zuB) FAILED", tag, bytes);
            db_signal(ctx, DB_ARR_AR_STAGE, seq, group.data(), static_cast<size_t>(members));
            db_signal(ctx, DB_ARR_AR_P1, seq, rev_partners.data(), rev_partners.size());
            db_signal(ctx, DB_ARR_AR_P2, seq, rev_partners.data(), rev_partners.size());
            return KuCCL_FAIL;
        }
    }
    /* stagings visible for ①: doorbell — readers are my group members only.
     * Signal them (one remote store each), then spin locally for theirs. */
    db_signal(ctx, DB_ARR_AR_STAGE, seq, group.data(), static_cast<size_t>(members));
    db_wait_local(ctx, DB_ARR_AR_STAGE, group.data(), static_cast<size_t>(members), seq);

    size_t my_start = 0, my_len = 0;
    /* Uneven spread: members past M (my_idx ≥ slices) own NO slice — every
     * phase skips and their segments are filled by ③ fanout. ar_hier_slice
     * is only defined for idx < m. */
    if (my_idx < slices)
        ar_hier_slice(params->count, my_idx, slices, &my_start, &my_len);
    const size_t my_off = my_start * dt_size;

    /* ① intra-group reduce-scatter: my slice ← every group peer's staged
     * data. Strip-parallel (dst-disjoint); the seed reads the user buffer
     * directly — no self round-trip through the window. */
    if (my_len > 0) {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_DATA_MOVE], g_timing.cnt[PHASE_DATA_MOVE]);
        ArFoldCtx fold;
        fold.dt = params->datatype;
        fold.op = params->redop;
        fold.dt_size = dt_size;
        fold.dst = ctx->recv_base;
        fold.self_src = static_cast<const char *>(params->sendbuff);
        fold.seg_ptrs = ctx->send_ptrs.data();
        fold.peers = group.data();
        fold.num_peers = members;
        fold.my_rank = comm->rank;
        fold.range_off = my_off;
        fold.count = my_len;
        fold.strips = std::min(g_config.parallel_threads,
                               static_cast<int>(std::max<size_t>(fold.count, 1))); /* no empty strips */
        if (!ar_fold_run(fold, comm->rank)) {
            detail::err_record("kupl allreduce(%s): fold① seed memcpy(%zuB) FAILED", tag, my_len * dt_size);
            db_signal(ctx, DB_ARR_AR_P1, seq, rev_partners.data(), rev_partners.size());
            db_signal(ctx, DB_ARR_AR_P2, seq, rev_partners.data(), rev_partners.size());
            return KuCCL_FAIL;
        }
    }
    /* ① partial visible to ② readers: doorbell instead of a full barrier —
     * the only readers are the K−1 partner owners (my_len==0 members have no
     * partial and nobody waits on them). Set-then-wait keeps both sides
     * progressing without deadlock. */
    if (my_len > 0) {
        db_signal(ctx, DB_ARR_AR_P1, seq, rev_partners.data(), rev_partners.size());
        db_wait_local(ctx, DB_ARR_AR_P1, partners.data(), partners.size(), seq);
    }

    /* ② o/e sub-block exchange: fold each partner's partial at MY sub-block
     * into my own sub-block in place (self_src = nullptr → strips fold into
     * the ① partial). Index-disjoint by construction: every owner reads each
     * remote segment only at its own sub-block index. */
    if (my_len > 0) {
        size_t sb_start = 0, sb_len = 0;
        ar_hier_slice(my_len, my_group, ngroups, &sb_start, &sb_len);
        if (sb_len > 0) {
            TimingScope ts(timing_enabled(), g_timing.ns[PHASE_DATA_MOVE], g_timing.cnt[PHASE_DATA_MOVE]);
            ArFoldCtx fold;
            fold.dt = params->datatype;
            fold.op = params->redop;
            fold.dt_size = dt_size;
            fold.dst = ctx->recv_base;
            fold.self_src = nullptr; /* in-place fold into the ① partial */
            fold.seg_ptrs = ctx->recv_ptrs.data();
            fold.peers = partners.data();
            fold.num_peers = static_cast<int>(partners.size());
            fold.my_rank = comm->rank; /* partners never contain self */
            fold.range_off = my_off + sb_start * dt_size;
            fold.count = sb_len;
            fold.strips = std::min(g_config.parallel_threads,
                                   static_cast<int>(std::max<size_t>(fold.count, 1))); /* no empty strips */
            if (!ar_fold_run(fold, comm->rank)) {
                detail::err_record("kupl allreduce(%s): fold② FAILED", tag);
                db_signal(ctx, DB_ARR_AR_P2, seq, rev_partners.data(), rev_partners.size());
                return KuCCL_FAIL;
            }
        }
    }
    /* ② sub-block globals visible to ③ fetchers: doorbell, waiters = the
     * partner owners that ③ reads from. */
    if (my_len > 0) {
        db_signal(ctx, DB_ARR_AR_P2, seq, rev_partners.data(), rev_partners.size());
        db_wait_local(ctx, DB_ARR_AR_P2, partners.data(), partners.size(), seq);
    }

    /* ③ owner fanout: fetch the foreign sub-block globals (completing my
     * slice), then push the full global slice to every group member. One
     * writer per slice per group; members idle until the recycle fence.
     * Partner[t] is the slice owner of remote group (t < my_group ? t : t+1)
     * — topo_build/socket_build push partners in ascending group order. */
    if (my_len > 0) {
        {
            TimingScope ts(timing_enabled(), g_timing.ns[PHASE_DATA_MOVE], g_timing.cnt[PHASE_DATA_MOVE]);
            for (size_t t = 0; t < partners.size(); t++) {
                const int pj = static_cast<int>(t) < my_group ? static_cast<int>(t) : static_cast<int>(t) + 1;
                size_t p_start = 0, p_len = 0;
                ar_hier_slice(my_len, pj, ngroups, &p_start, &p_len);
                if (p_len == 0)
                    continue;
                const size_t off = my_off + p_start * dt_size;
                if (kupl_memcpy(ctx->recv_base + off, ctx->recv_ptrs[partners[t]] + off, p_len * dt_size) != KUPL_OK) {
                    detail::err_record("kupl allreduce(%s): fetch memcpy(%zuB) FAILED", tag, p_len * dt_size);
                    return KuCCL_FAIL;
                }
            }
        }
        {
            TimingScope ts(timing_enabled(), g_timing.ns[PHASE_DATA_MOVE], g_timing.cnt[PHASE_DATA_MOVE]);
            ArPushCtx push;
            push.recv_ptrs = ctx->recv_ptrs.data();
            push.group = group.data();
            push.my_rank = comm->rank;
            push.src = ctx->recv_base + my_off;
            push.dst_off = my_off;
            push.bytes = my_len * dt_size;
            const bool par = g_config.parallel_threads > 1 && bytes >= g_config.parallel_min_bytes;
            const int64_t tasks = members; /* one per member; self skipped in-task */
            bool ran = false;
            if (par)
                ran = run_parallel(tasks, ar_push_task, &push, comm->rank);
            if (!ran)
                run_serial(tasks, ar_push_task, &push);
            if (push.failed) {
                detail::err_record("kupl allreduce(%s): fanout memcpy FAILED", tag);
                return KuCCL_FAIL;
            }
        }
    }

    timed_fence(ctx->recv_win, comm->rank); /* recycle: ③ writes visible for
                                               copy-backs; next round's ① waits
                                               out this round's ③ reads */

    priv->user_dst = params->recvbuff;
    priv->win_src = ctx->recv_base;
    priv->copy_bytes = bytes;
    KuCCL_LOG_TRACE(comm->rank, "kupl allreduce(%s): done (members=%d slices=%d groups=%d bytes=%zuB)", tag, members,
                    slices, ngroups, bytes);
    return KuCCL_SUCCESS;
}

static kucclResult_t allreduce_hier_die(kucclComm *comm, KuplComm *ctx, const CollParams *params, size_t dt_size,
                                        size_t bytes, KuplReq *priv)
{
    return allreduce_layered(comm, ctx, params, dt_size, bytes, priv, ctx->topo.local_ranks, ctx->topo.partners,
                             ctx->topo.rev_partners, ctx->topo.my_node, ctx->topo.groups, "hier_die");
}

static kucclResult_t allreduce_hier_socket(kucclComm *comm, KuplComm *ctx, const CollParams *params, size_t dt_size,
                                           size_t bytes, KuplReq *priv)
{
    return allreduce_layered(comm, ctx, params, dt_size, bytes, priv, ctx->topo.socket_local_ranks,
                             ctx->topo.socket_partners, ctx->topo.socket_rev_partners, ctx->topo.my_socket,
                             ctx->topo.socket_groups, "hier_socket");
}

static kucclResult_t impl_allreduce(kucclComm *comm, KuplComm *ctx, const CollParams *params, size_t dt_size,
                                    KuplReq *priv, kupl_shm_request_h *native_req)
{
    TimingScope t_total(timing_enabled(), g_timing.ns_op[OP_ALLREDUCE], g_timing.cnt_op[OP_ALLREDUCE]);
    const size_t bytes = params->count * dt_size;
    if (bytes == 0 || bytes > static_cast<size_t>(INT32_MAX)) {
        detail::err_record("kupl allreduce: invalid bytes=%zu (count=%zu dt_size=%zu)", bytes, params->count, dt_size);
        return KuCCL_FAIL;
    }
    *native_req = nullptr;

    /* Two exposed algorithms: flat (default/unset) and hier (forced, any
     * size); auto is size-routed — the layered path picks its granularity
     * by topology and degrades to flat on flat layouts. */
    const AllreduceAlgo ar = g_config.ar_route(bytes, ctx->topo.usable, ctx->topo.socket_usable);
    if (ar != ALLREDUCE_FLAT) {
        KuCCL_LOG_TRACE(comm->rank, "[AR-route] bytes=%zuB ws=%d → %s", bytes, comm->nranks, ar_name(ar));
        return ar == ALLREDUCE_HIER_SOCKET ? allreduce_hier_socket(comm, ctx, params, dt_size, bytes, priv) :
                                             allreduce_hier_die(comm, ctx, params, dt_size, bytes, priv);
    }

    if (params->redop == kucclSum)
        return allreduce_sum_native(comm, ctx, params, bytes, priv, native_req);

    /* MIN/MAX: RH runs entirely in the recv window (it stages user data
     * itself); direct needs send (peers read raw data) + recv accumulator */
    const bool use_recursive = is_pow2(comm->nranks) && comm->nranks > 4;
    drain_peer_acks(comm, ctx);
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_SCRATCH], g_timing.cnt[PHASE_SCRATCH]);
        if (ensure_recv(ctx, bytes) != KuCCL_SUCCESS)
            return KuCCL_FAIL;
        if (!use_recursive && ensure_send(ctx, bytes) != KuCCL_SUCCESS)
            return KuCCL_FAIL;
    }
    if (use_recursive) {
        const kucclResult_t ret = allreduce_minmax_recursive(comm, ctx, params, dt_size, bytes, priv);
        if (ret == KuCCL_SUCCESS)
            KuCCL_LOG_TRACE(comm->rank, "kupl allreduce MIN/MAX(RH-RD): done (N=%d, count=%zu)", comm->nranks,
                            params->count);
        return ret;
    }
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_STAGE_OUT], g_timing.cnt[PHASE_STAGE_OUT]);
        if (kupl_memcpy(ctx->send_base, params->sendbuff, bytes) != KUPL_OK) {
            detail::err_record("kupl allreduce(MIN/MAX): stage memcpy(%zuB) FAILED", bytes);
            return KuCCL_FAIL;
        }
    }
    const kucclResult_t ret = allreduce_minmax_direct(comm, ctx, params, bytes, priv);
    if (ret == KuCCL_SUCCESS)
        KuCCL_LOG_TRACE(comm->rank, "kupl allreduce MIN/MAX(direct): done (N=%d)", comm->nranks);
    return ret;
}

static kucclResult_t impl_broadcast(kucclComm *comm, KuplComm *ctx, const CollParams *params, size_t dt_size,
                                    KuplReq *priv, kupl_shm_request_h *native_req)
{
    TimingScope t_total(timing_enabled(), g_timing.ns_op[OP_BROADCAST], g_timing.cnt_op[OP_BROADCAST]);
    const size_t bytes = params->count * dt_size;
    if (bytes == 0 || bytes > static_cast<size_t>(INT32_MAX) || params->root < 0 || params->root >= comm->nranks) {
        detail::err_record("kupl bcast: invalid bytes=%zu root=%d (nranks=%d)", bytes, params->root, comm->nranks);
        return KuCCL_FAIL;
    }
    *native_req = nullptr;

    if (g_config.bcast_native) {
        drain_peer_acks(comm, ctx);
        {
            TimingScope ts(timing_enabled(), g_timing.ns[PHASE_SCRATCH], g_timing.cnt[PHASE_SCRATCH]);
            if (ensure_recv(ctx, bytes) != KuCCL_SUCCESS)
                return KuCCL_FAIL;
        }
        {
            TimingScope ts(timing_enabled(), g_timing.ns[PHASE_STAGE_OUT], g_timing.cnt[PHASE_STAGE_OUT]);
            if (comm->rank == params->root && kupl_memcpy(ctx->recv_base, params->sendbuff, bytes) != KUPL_OK) {
                detail::err_record("kupl bcast(native): stage memcpy(%zuB) FAILED", bytes);
                return KuCCL_FAIL;
            }
        }
        {
            TimingScope ts(timing_enabled(), g_timing.ns[PHASE_NATIVE], g_timing.cnt[PHASE_NATIVE]);
            if (kupl_shm_bcast_init(ctx->recv_base, static_cast<int>(params->count), map_datatype(params->datatype),
                                    params->root, ctx->comm, native_req) != 0) {
                detail::err_record("kupl bcast(native): init FAILED");
                return KuCCL_FAIL;
            }
            if (kupl_shm_request_start(*native_req) != 0) {
                detail::err_record("kupl bcast(native): start FAILED");
                kupl_shm_request_free(*native_req);
                *native_req = nullptr;
                return KuCCL_FAIL;
            }
        }
        priv->user_dst = params->recvbuff;
        priv->win_src = ctx->recv_base;
        priv->copy_bytes = bytes;
        return KuCCL_SUCCESS;
    }

    /* direct pull via the flag protocol (counters/drain shared with allgather) */
    drain_peer_acks(comm, ctx);
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_SCRATCH], g_timing.cnt[PHASE_SCRATCH]);
        if (ctx->send_cap < bytes * 2 && ensure_send(ctx, bytes * 2) != KuCCL_SUCCESS)
            return KuCCL_FAIL;
    }
    const uint64_t round = ctx->flag_round + 1;
    const size_t off = static_cast<size_t>(round & 1) * (ctx->send_cap / 2); /* same layout as the flag path */

    if (comm->rank == params->root) {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_STAGE_OUT], g_timing.cnt[PHASE_STAGE_OUT]);
        if (kupl_memcpy(ctx->send_base + off, params->sendbuff, bytes) != KUPL_OK) {
            detail::err_record("kupl bcast(direct): stage memcpy(%zuB) FAILED", bytes);
            abort_flag_round(ctx, round);
            return KuCCL_FAIL;
        }
        __atomic_store_n(ctx->ready_flag, static_cast<int64_t>(round), __ATOMIC_RELEASE); /* data before bell */
    } else {
        {
            TimingScope tw(timing_enabled(), g_timing.ns[PHASE_FLAG_WAIT], g_timing.cnt[PHASE_FLAG_WAIT]);
            spin_wait(ctx->peer_ready[params->root], round);
        }
        {
            TimingScope ts(timing_enabled(), g_timing.ns[PHASE_DATA_MOVE], g_timing.cnt[PHASE_DATA_MOVE]);
            if (kupl_memcpy(params->recvbuff, ctx->send_ptrs[params->root] + off, bytes) != KUPL_OK) {
                detail::err_record("kupl bcast(direct): read memcpy(%zuB) FAILED", bytes);
                abort_flag_round(ctx, round);
                return KuCCL_FAIL;
            }
        }
    }
    __atomic_store_n(ctx->ack_flag, static_cast<int64_t>(round), __ATOMIC_RELEASE);
    ctx->flag_round = round;

    /* data already in the user buffer */
    priv->user_dst = nullptr;
    priv->win_src = nullptr;
    priv->copy_bytes = 0;
    KuCCL_LOG_TRACE(comm->rank, "kupl bcast(direct): done (round=%llu, %zuB)", static_cast<unsigned long long>(round),
                    bytes);
    return KuCCL_SUCCESS;
}

/* ── AllToAll direct pull ──
 *
 * No sharing (each (sender, receiver) chunk is unique to that pair), no
 * reduction — every chunk crosses the boundary exactly once, which is the
 * crossing-traffic lower bound. Design: stage the full send once, one
 * fence, then every rank pulls its chunk from each peer's staged send
 * segment straight into the user recvbuf — no native call, no copy-back,
 * 2 fences total. Only own recvbuf is written; the trailing fence orders
 * next staging after this round's peer reads. Sender s's chunk for me sits
 * at the same offset in its segment (both layouts are [nranks][per_rank]). */
struct A2aPullCtx {
    char *const *send_ptrs = nullptr; /* peer send segment bases */
    const char *self_src = nullptr;   /* user sendbuff (self chunk: no window round-trip) */
    char *recvbuff = nullptr;         /* user recvbuf base */
    int my_rank = 0;
    size_t chunk = 0; /* per-rank bytes */
    volatile int failed = 0;
};

static void a2a_pull_task(kupl_nd_range_t *nd, void *args, int, int)
{
    auto *p = static_cast<A2aPullCtx *>(args);
    const kupl_range_t &range = nd->nd_range[0];
    /* Receiver j pulls sender s's block ADDRESSED TO j. The staged layout is
     * [nranks][chunk], so that block lives at offset j·chunk (my_rank), not
     * s·chunk (s's own self block) — the transpose is the point of alltoall. */
    const size_t src_off = static_cast<size_t>(p->my_rank) * p->chunk;
    for (int64_t r = range.lower; r < range.upper && !p->failed; r += range.step) {
        const size_t dst_off = static_cast<size_t>(r) * p->chunk;
        const char *src = (r == p->my_rank) ? p->self_src + src_off : p->send_ptrs[r] + src_off;
        if (kupl_memcpy(p->recvbuff + dst_off, src, p->chunk) != KUPL_OK) {
            p->failed = 1;
            return;
        }
    }
}

static kucclResult_t impl_alltoall(kucclComm *comm, KuplComm *ctx, const CollParams *params, size_t dt_size,
                                   KuplReq *priv, kupl_shm_request_h *native_req)
{
    TimingScope t_total(timing_enabled(), g_timing.ns_op[OP_ALLTOALL], g_timing.cnt_op[OP_ALLTOALL]);
    const size_t per_rank = params->count * dt_size;
    const size_t total = per_rank * static_cast<size_t>(comm->nranks);
    if (per_rank == 0 || total > static_cast<size_t>(INT32_MAX)) {
        detail::err_record("kupl alltoall: invalid per_rank=%zu total=%zu", per_rank, total);
        return KuCCL_FAIL;
    }

    /* direct (default) vs native (opt-in, A/B) */
    if (g_config.a2a_direct) {
        drain_peer_acks(comm, ctx);
        {
            TimingScope ts(timing_enabled(), g_timing.ns[PHASE_SCRATCH], g_timing.cnt[PHASE_SCRATCH]);
            /* recv window unused: pulls land straight in the user recvbuf */
            if (ensure_send(ctx, total) != KuCCL_SUCCESS)
                return KuCCL_FAIL;
        }
        {
            TimingScope ts(timing_enabled(), g_timing.ns[PHASE_STAGE_OUT], g_timing.cnt[PHASE_STAGE_OUT]);
            /* alltoall stages the FULL [nranks][chunk] matrix (pull model needs
             * every receiver's block resident) — the largest stage of all ops,
             * so grain-parallel copying pays off earlier than in allgather */
            if (!stage_copy(ctx->send_base, static_cast<const char *>(params->sendbuff), total, comm->rank)) {
                detail::err_record("kupl alltoall(direct): stage memcpy(%zuB) FAILED", total);
                return KuCCL_FAIL;
            }
        }
        timed_fence(ctx->send_win, comm->rank); /* stagings visible for the pulls */
        /* Optional int32 payload dump after staging. Values depend on the
         * caller's payload; zero alone is not evidence of a visibility error. */
        if (std::getenv("KUCCL_KUPL_A2A_DEBUG") != nullptr && comm->nranks > 1 && dt_size == sizeof(int32_t)) {
            const int32_t mine = *reinterpret_cast<volatile const int32_t *>(
                ctx->send_base + static_cast<size_t>(comm->rank) * per_rank);
            const int32_t p1 = *reinterpret_cast<volatile const int32_t *>(ctx->send_ptrs[1] +
                                                                           static_cast<size_t>(comm->rank) * per_rank);
            KuCCL_LOG_INFO(comm->rank, "[a2a-debug] my_base=%p p0=%p p1=%p mine@my_slot=%d peer1@my_slot=%d",
                           static_cast<const void *>(ctx->send_base), static_cast<const void *>(ctx->send_ptrs[0]),
                           static_cast<const void *>(ctx->send_ptrs[1]), mine, p1);
            /* /proc/self/maps line backing each cached peer base — the shm FILE
             * name exposes self-aliasing (own file mapped for every peer slot) */
            for (int peer = 0; peer < comm->nranks && peer < 3; peer++) {
                char line[512] = {};
                const uintptr_t addr = reinterpret_cast<uintptr_t>(ctx->send_ptrs[peer]);
                if (FILE *fp = std::fopen("/proc/self/maps", "r"); fp != nullptr) {
                    char raw[512];
                    while (std::fgets(raw, sizeof(raw), fp) != nullptr) {
                        uintptr_t lo = 0, hi = 0;
                        if (std::sscanf(raw, "%lx-%lx", &lo, &hi) == 2 && addr >= lo && addr < hi) {
                            std::snprintf(line, sizeof(line), "%s", raw);
                            break;
                        }
                    }
                    std::fclose(fp);
                }
                for (char *nl = line; *nl != '\0'; nl++)
                    if (*nl == '\n')
                        *nl = ' ';
                KuCCL_LOG_INFO(comm->rank, "[a2a-debug] maps[%d]: %s", peer, line);
            }
        }
        {
            TimingScope ts(timing_enabled(), g_timing.ns[PHASE_DATA_MOVE], g_timing.cnt[PHASE_DATA_MOVE]);
            A2aPullCtx pull;
            pull.send_ptrs = ctx->send_ptrs.data();
            pull.self_src = static_cast<const char *>(params->sendbuff);
            pull.recvbuff = static_cast<char *>(params->recvbuff);
            pull.my_rank = comm->rank;
            pull.chunk = per_rank;
            const bool par = g_config.parallel_threads > 1 && per_rank >= g_config.parallel_min_bytes;
            const int64_t tasks = comm->nranks; /* one per rank; self handled in-task */
            bool ran = false;
            if (par)
                ran = run_parallel(tasks, a2a_pull_task, &pull, comm->rank);
            if (!ran)
                run_serial(tasks, a2a_pull_task, &pull);
            if (pull.failed) {
                detail::err_record("kupl alltoall(direct): pull memcpy FAILED");
                return KuCCL_FAIL;
            }
        }
        timed_fence(ctx->send_win, comm->rank); /* recycle: stragglers finished pulling my
                                                  staged chunks before the next staging */
        priv->user_dst = nullptr;               /* data already in the user recvbuff */
        priv->win_src = nullptr;
        priv->copy_bytes = 0;
        KuCCL_LOG_TRACE(comm->rank, "kupl alltoall(direct): done (nranks=%d per_rank=%zuB)", comm->nranks, per_rank);
        return KuCCL_SUCCESS;
    }

    drain_peer_acks(comm, ctx);
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_SCRATCH], g_timing.cnt[PHASE_SCRATCH]);
        if (ensure_send(ctx, total) != KuCCL_SUCCESS || ensure_recv(ctx, total) != KuCCL_SUCCESS)
            return KuCCL_FAIL;
    }
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_STAGE_OUT], g_timing.cnt[PHASE_STAGE_OUT]);
        if (!stage_copy(ctx->send_base, static_cast<const char *>(params->sendbuff), total, comm->rank)) {
            detail::err_record("kupl alltoall: stage memcpy(%zuB) FAILED", total);
            return KuCCL_FAIL;
        }
    }
    {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_NATIVE], g_timing.cnt[PHASE_NATIVE]);
        /* diagnostics: dump recv-window bases — physical collapse/stride
         * anomalies here explain native-path zero data */
        {
            void *q0 = nullptr, *qlast = nullptr;
            kupl_shm_win_query(ctx->recv_win, 0, &q0);
            kupl_shm_win_query(ctx->recv_win, comm->nranks - 1, &qlast);
            KuCCL_LOG_INFO(comm->rank,
                           "kupl alltoall(native) window dump: recv_base=%p q0=%p qlast=%p stride=%td total=%zu",
                           (const void *)ctx->recv_base, q0, qlast, (char *)qlast - (char *)q0, total);
        }
        if (kupl_shm_alltoall_init(ctx->send_base, static_cast<int>(params->count), map_datatype(params->datatype),
                                   ctx->recv_base, static_cast<int>(params->count), map_datatype(params->datatype),
                                   ctx->comm, native_req) != 0) {
            detail::err_record("kupl alltoall: init FAILED");
            return KuCCL_FAIL;
        }
        if (kupl_shm_request_start(*native_req) != 0) {
            detail::err_record("kupl alltoall: start FAILED");
            kupl_shm_request_free(*native_req);
            *native_req = nullptr;
            return KuCCL_FAIL;
        }
    }
    priv->user_dst = params->recvbuff;
    priv->win_src = ctx->recv_base;
    priv->copy_bytes = total;
    return KuCCL_SUCCESS;
}

static kucclResult_t kupl_process_init(kucclComm *)
{
    /* no process-wide setup: kupl_shm_init happens lazily inside
     * kupl_shm_comm_create */
    return KuCCL_SUCCESS;
}

static void kupl_process_finalize() {}

/* Notify window + NUMA exchange + flag caches, all in one shot:
 * after this returns, hot paths never call win_query again. */
static kucclResult_t notify_and_topo_init(KuplComm *ctx, int rank)
{
    void *base = nullptr;
    kupl_shm_win_h win = nullptr;
    if (kupl_shm_win_alloc(1024, ctx->comm, &base, &win) != 0 || base == nullptr || win == nullptr) {
        detail::err_record("kupl: notify win_alloc FAILED");
        return KuCCL_FAIL;
    }
    ctx->notify_win = win;
    ctx->notify_base = static_cast<char *>(base);
    ctx->ready_flag = reinterpret_cast<volatile int64_t *>(base);
    ctx->ack_flag = reinterpret_cast<volatile int64_t *>(ctx->notify_base + 64);
    *ctx->ready_flag = 0;
    *ctx->ack_flag = 0;
    *reinterpret_cast<volatile int32_t *>(ctx->notify_base + 72) = static_cast<int32_t>(detect_numa_node());
    *reinterpret_cast<volatile int32_t *>(ctx->notify_base + 76) = static_cast<int32_t>(detect_socket_id());

    /* publish node + socket ids, then cache every peer's flags and read them back.
     * Self-built mappings (same defect bypass as the data windows): the
     * doorbell flags AND the topology probe both read peer segments here —
     * a self-aliased mapping would silently kill doorbell ordering and
     * flatten the topology to "everyone on my node". */
    kupl_shm_fence(win);
    const int n = ctx->nranks;
    ctx->peer_ready.assign(n, nullptr);
    ctx->peer_ack.assign(n, nullptr);
    std::vector<int> node_of(static_cast<size_t>(n), -1);
    std::vector<int> socket_of(static_cast<size_t>(n), -1);
    std::vector<char *> segs;
    if (g_config.peer_map_own && ctx->owner != nullptr) {
        if (self_peer_map(ctx, ctx->notify_base, 1024, segs, "notify") != KuCCL_SUCCESS)
            return KuCCL_FAIL;
    } else {
        segs.assign(static_cast<size_t>(n), nullptr);
        for (int peer = 0; peer < n; peer++) {
            void *pbase = nullptr;
            if (kupl_shm_win_query(win, peer, &pbase) != 0 || pbase == nullptr) {
                detail::err_record("kupl: notify win_query(rank=%d) FAILED", peer);
                return KuCCL_FAIL;
            }
            segs[static_cast<size_t>(peer)] = static_cast<char *>(pbase);
        }
    }
    for (int peer = 0; peer < n; peer++) {
        auto *seg = segs[static_cast<size_t>(peer)];
        ctx->peer_ready[peer] = reinterpret_cast<volatile int64_t *>(seg);
        ctx->peer_ack[peer] = reinterpret_cast<volatile int64_t *>(seg + 64);
        node_of[static_cast<size_t>(peer)] = *reinterpret_cast<volatile int32_t *>(seg + 72);
        socket_of[static_cast<size_t>(peer)] = *reinterpret_cast<volatile int32_t *>(seg + 76);
    }
    topo_build(ctx->topo, node_of, rank);
    socket_build(ctx->topo, socket_of, rank);
    return KuCCL_SUCCESS;
}

static void kupl_group_fini(kucclComm *comm)
{
    kupl_timing_report(comm ? comm->rank : -1, comm ? comm->nranks : 0);
    auto *ctx = static_cast<KuplComm *>(comm->be_ctx);
    if (ctx == nullptr)
        return;
    if (ctx->send_win != nullptr) {
        kupl_shm_win_free(ctx->send_win);
        ctx->send_win = nullptr;
    }
    if (ctx->recv_win != nullptr) {
        kupl_shm_win_free(ctx->recv_win);
        ctx->recv_win = nullptr;
    }
    if (ctx->notify_win != nullptr) {
        kupl_shm_win_free(ctx->notify_win);
        ctx->notify_win = nullptr;
    }
    if (ctx->comm != nullptr) {
        kupl_shm_comm_destroy(ctx->comm);
        ctx->comm = nullptr;
    }
    delete ctx;
    comm->be_ctx = nullptr;
}

static kucclResult_t kupl_group_init(kucclComm *comm)
{
    auto *ctx = new (std::nothrow) KuplComm();
    if (ctx == nullptr) {
        detail::err_record("kupl: alloc group ctx FAILED");
        return KuCCL_FAIL;
    }
    ctx->owner = comm;
    ctx->nranks = comm->nranks;
    comm->be_ctx = ctx;

    /* comm_create lazily runs kupl_shm_init and collects group info through
     * the injected OOB callbacks (cookie = ctx) */
    kupl_shm_oob_cb_t cbs;
    cbs.oob_allgather = kupl_oob_allgather_cb;
    cbs.oob_barrier = kupl_oob_barrier_cb;
    if (kupl_shm_comm_create(comm->nranks, comm->rank, static_cast<int>(::getpid()), &cbs, ctx, &ctx->comm) != 0 ||
        ctx->comm == nullptr) {
        detail::err_record("kupl: kupl_shm_comm_create FAILED");
        KuCCL_LOG_ERROR(comm->rank, "kupl_shm_comm_create FAILED");
        delete ctx;
        comm->be_ctx = nullptr;
        return KuCCL_FAIL;
    }

    if (notify_and_topo_init(ctx, comm->rank) != KuCCL_SUCCESS) {
        kupl_group_fini(comm);
        return KuCCL_FAIL;
    }
    KuCCL_LOG_INFO(comm->rank, "kupl shm comm created (nranks=%d, topo=%s, nodes=%d, socket_pair=%s)", comm->nranks,
                   ctx->topo.usable ? "pair-exchange" : "flat",
                   ctx->topo.usable ? static_cast<int>(ctx->topo.partners.size() + 1) : 1,
                   ctx->topo.socket_usable ? "socket-pair" : "off");
    return KuCCL_SUCCESS;
}

static kucclResult_t kupl_coll_start(kucclComm *comm, CollOp op, const CollParams *params, kucclRequest *req)
{
    auto *ctx = static_cast<KuplComm *>(comm->be_ctx);
    if (ctx == nullptr) {
        detail::err_record("kupl: comm not initialized");
        return KuCCL_FAIL;
    }
    if (ctx->broken) {
        detail::err_record("kupl: comm unusable (window cache failed earlier)");
        return KuCCL_FAIL;
    }

    const size_t dt_size = kupl_dt_size(params->datatype);
    if (dt_size == 0 || params->count == 0 || params->count > SIZE_MAX / dt_size) {
        detail::err_record("kupl: invalid params (datatype=%d count=%zu)", static_cast<int>(params->datatype),
                           params->count);
        return KuCCL_FAIL;
    }

    /* unique_ptr ownership: every early return cleans up automatically;
     * flag rounds are advanced inside the implementations on failure */
    std::unique_ptr<KuplReq> priv(new (std::nothrow) KuplReq());
    if (priv == nullptr) {
        detail::err_record("kupl: alloc request state FAILED");
        return KuCCL_FAIL;
    }

    kupl_shm_request_h native_req = nullptr;
    kucclResult_t ret = KuCCL_FAIL;
    switch (op) {
        case CollOp::AllGather:
            ret = impl_allgather(comm, ctx, params, dt_size, priv.get(), &native_req);
            break;
        case CollOp::AllReduce:
            ret = impl_allreduce(comm, ctx, params, dt_size, priv.get(), &native_req);
            break;
        case CollOp::Broadcast:
            ret = impl_broadcast(comm, ctx, params, dt_size, priv.get(), &native_req);
            break;
        case CollOp::AllToAll:
            ret = impl_alltoall(comm, ctx, params, dt_size, priv.get(), &native_req);
            break;
    }
    if (ret != KuCCL_SUCCESS)
        return ret;

    priv->req = native_req;
    req->priv = priv.release();
    return KuCCL_SUCCESS;
}

/* kupl request_start already drove the data plane synchronously: copy back
 * (if staged), release the kupl request, report COMPLETE. */
static kucclResult_t kupl_req_test(kucclRequest *req, int *flag, kucclStatus_t *status)
{
    auto *priv = static_cast<KuplReq *>(req->priv);
    if (priv == nullptr) {
        *flag = 1;
        *status = KuCCL_STATUS_ERROR;
        return KuCCL_SUCCESS;
    }
    if (priv->user_dst != nullptr && priv->copy_bytes > 0) {
        TimingScope ts(timing_enabled(), g_timing.ns[PHASE_COPY_BACK], g_timing.cnt[PHASE_COPY_BACK]);
        if (kupl_memcpy(priv->user_dst, priv->win_src, priv->copy_bytes) != KUPL_OK) {
            detail::err_record("kupl: copy-back memcpy(%zuB) FAILED", priv->copy_bytes);
            if (priv->req != nullptr)
                kupl_shm_request_free(priv->req);
            delete priv;
            req->priv = nullptr;
            *flag = 1;
            *status = KuCCL_STATUS_ERROR;
            return KuCCL_SUCCESS;
        }
    }
    if (priv->req != nullptr)
        kupl_shm_request_free(priv->req);
    delete priv;
    req->priv = nullptr;
    *flag = 1;
    *status = KuCCL_STATUS_COMPLETE;
    return KuCCL_SUCCESS;
}

static void kupl_req_cancel(kucclRequest *req)
{
    auto *priv = static_cast<KuplReq *>(req->priv);
    if (priv == nullptr)
        return;
    /* kupl has no cancel semantics: start is already synchronous — just
     * reclaim the request */
    if (priv->req != nullptr)
        kupl_shm_request_free(priv->req);
    delete priv;
    req->priv = nullptr;
}

static void kupl_comm_purge(kucclComm *) {}

static kucclResult_t kupl_flush(kucclComm *)
{
    /* shm data plane is synchronously complete — nothing in flight */
    return KuCCL_SUCCESS;
}

} /* anonymous namespace */

const Ops *kupl_ops()
{
    static const Ops ops = {
        "kupl",           &kupl_process_init, &kupl_process_finalize, &kupl_group_init, &kupl_group_fini,
        &kupl_coll_start, &kupl_req_test,     &kupl_req_cancel,       &kupl_comm_purge, &kupl_flush,
    };
    return &ops;
}

} /* namespace backend */
} /* namespace kuccl */

#else /* !KuCCL_HAVE_KUPL — stub (built when KUPL was not found) */

namespace kuccl {
namespace backend {

namespace {

kucclResult_t kupl_unavailable(const char *name)
{
    detail::err_record("kupl backend: %s unavailable (built without KuCCL_HAVE_KUPL; "
                       "set KUPL_DIR to a KUPL install prefix and rebuild)",
                       name);
    return KuCCL_FAIL;
}

kucclResult_t kupl_process_init(kucclComm *comm)
{
    KuCCL_LOG_ERROR(comm->rank, "KUCCL_BACKEND_TYPE=kupl: this build is not linked against KUPL"
                                " (rebuild with KUPL_DIR pointing to a KUPL install prefix);"
                                " available backend: xucg.");
    return kupl_unavailable("process_init");
}

void kupl_process_finalize() {}

kucclResult_t kupl_group_init(kucclComm *)
{
    return kupl_unavailable("group_init");
}
void kupl_group_fini(kucclComm *) {}

kucclResult_t kupl_coll_start(kucclComm *, CollOp, const CollParams *, kucclRequest *)
{
    return kupl_unavailable("collective start");
}

kucclResult_t kupl_req_test(kucclRequest *, int *, kucclStatus_t *)
{
    return kupl_unavailable("request test");
}

void kupl_req_cancel(kucclRequest *) {}

void kupl_comm_purge(kucclComm *) {}

kucclResult_t kupl_flush(kucclComm *)
{
    return kupl_unavailable("flush");
}

} /* anonymous namespace */

const Ops *kupl_ops()
{
    static const Ops ops = {
        "kupl",           &kupl_process_init, &kupl_process_finalize, &kupl_group_init, &kupl_group_fini,
        &kupl_coll_start, &kupl_req_test,     &kupl_req_cancel,       &kupl_comm_purge, &kupl_flush,
    };
    return &ops;
}

} /* namespace backend */
} /* namespace kuccl */

#endif /* KuCCL_HAVE_KUPL */
