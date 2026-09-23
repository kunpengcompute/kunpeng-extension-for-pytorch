/*
 * xucg 后端 —— 基于 UCG 公有接口的集合通信实现（默认后端）
 *
 * 后端边界（见 backend.hpp 的约定）：
 *   - 集合请求：ucg_request_*_init → ucg_request_start → ucg_request_test
 *     推进用 ucg_progress（UCG 公有接口），状态码只判 UCG_OK / UCG_INPROGRESS；
 *   - UCP 资源层：ucg 的编程模型要求调用方注入 ucp context/worker/ep
 *     （ucg_global_init 的 oob_resource 回调），因此 ucp_init/ucp_ep_create
 *     等资源调用封闭在本文件内，对 kuccl 骨架（coll.cpp/comm.cpp）不可见。
 *
 * 核心机制：
 *   - 进程级单例（UCP/UCG context + 全局地址表）
 *   - 两阶段变长地址 allgather
 *   - 进程拓扑：OOB 交换主机哈希 → 节点表 + PPN，经 get_proc_info 的
 *     location.node_id 喂 UCG（节点感知算法与策略表分档的唯一拓扑入口；
 *     交换失败全组一致降级为平铺模型——UCG 要求全组同报或同不报）
 *   - FNV-1a group_id 哈希 + instance_token 防撞
 *   - UCG 回调（oob_cb / get_ucp_ep / get_ucp_worker/context / get_proc_info）
 */

#include "backend.hpp" /* err_record / g_api_lock（经 comm_impl.hpp）*/

#include <ucp/api/ucp.h>
#include <ucg/api/ucg.h>

#include <dirent.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <new>
#include <unistd.h>
#include <vector>

namespace kuccl {
namespace backend {

/* ════════════════════════════════════════════════════════════
 * 进程级共享资源（单例）
 * ════════════════════════════════════════════════════════════*/
namespace {

struct XucgGlobal {
    ucp_context_h ucp_context = nullptr;
    ucp_worker_h ucp_worker = nullptr;
    ucp_address_t *my_address = nullptr;
    size_t my_address_len = 0;
    ucg_context_h ucg_context = nullptr;

    int global_rank = -1;
    int global_nranks = 0;
    uint8_t *world_addrs = nullptr;
    size_t *world_addr_lens = nullptr;
    size_t max_addr_len = 0;
    /* 进程拓扑：主机哈希（OOB 交换）→ 节点表（node_id 按首次出现顺序编号）
     * 与本节点 PPN——经 get_proc_info 的 location.node_id 喂 UCG（算法选择
     * 的唯一拓扑入口；num_local_procs 仅作 UCX estimated_num_ppn 提示） */
    std::vector<uint64_t> host_hash;
    std::vector<int32_t> node_id;
    int32_t nnode = 0;              /* 节点数（topology_ok 时有效；诊断用） */
    std::vector<int32_t> socket_id; /* NUMA 节点号（socket_ok 时有效，稠密化） */
    int32_t nsocket = 0;
    bool socket_ok = false; /* NUMA 全组一致有效（任一 rank 探测失败则全组不喂） */
    uint32_t num_local_procs = 0;
    bool topology_ok = false; /* node_id 表有效且全组一致（失败全组降级平铺） */
    bool initialized = false;
};

XucgGlobal g_xucg;

/* UCG 拓扑消费计数（诊断）：get_proc_info 总调用数 / 其中带 node_id 上报的数。
 * ucg_group_create 返回后读一次——两个数相等且 >0 即证明 UCG 真的取走了拓扑 */
std::atomic<int> g_topo_queries{0};
std::atomic<int> g_topo_reported{0};

/* 后端私有通信域状态（挂在 kucclComm::be_ctx）。
 * 生命周期：process_init 分配（ucg_init 的 oob_group cookie 需要），
 * group_init 填充组信息，group_fini 释放。 */
struct XucgComm {
    ucg_group_h group;
    uint32_t group_id;
    ucg_rank_t *rank_map;
    uint32_t rank_map_size;
    kucclComm *owner; /* 反指宿主（UCG 回调的 group cookie 是本结构） */
};

/* 确保 comm 挂有后端私有状态（幂等；失败返回 nullptr 并记错误） */
XucgComm *xucg_ensure_ctx(kucclComm *comm)
{
    if (comm->be_ctx != nullptr)
        return static_cast<XucgComm *>(comm->be_ctx);
    auto *xctx = new (std::nothrow) XucgComm();
    if (xctx == nullptr) {
        detail::err_record("xucg: alloc group ctx FAILED");
        return nullptr;
    }
    xctx->group = nullptr;
    xctx->owner = comm;
    comm->be_ctx = xctx;
    return xctx;
}

/* 后端私有请求状态（挂在 kucclRequest::priv） */
struct XucgReq {
    ucg_request_h ucg_req = nullptr;
    void *persist_entry = nullptr; /* PersistentEntry*（未入池为 null） */
    uint32_t op_id = 0;            /* 剖析表下标 */
    uint32_t test_calls = 0;
    /* allgatherv/alltoallv 的 counts/displs 与 request_info：UCG 在 init 后
     * （start/progress 阶段）仍会读取这些指针，必须存活到请求生命周期结束，
     * 因此挂在 priv 上随请求析构，禁止用函数局部容器传给 init */
    std::vector<int32_t> keep_v[4];
    ucg_request_info_t keep_info{};
};

/* ucp_init 的 request_size 预留 */
struct kuccl_request_placeholder {
    char buf[160];
};

/* bootstrap 记录：一轮 OOB 交换 global_rank + instance_token + 主机哈希 */
struct XucgBootstrapRecord {
    int32_t global_rank;
    uint32_t reserved;
    uint64_t instance_token;
    uint64_t host_hash;
};
static_assert(sizeof(XucgBootstrapRecord) == 24, "unexpected layout");

/* SplitMix64 — instance_token 生成 */
uint64_t xucg_mix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/* FNV-1a feed helpers — group_id 哈希 */
void xucg_fnv1a_feed_u32(uint32_t *hash, uint32_t value)
{
    for (int b = 0; b < 4; ++b) {
        *hash ^= static_cast<uint8_t>(value & 0xffu);
        *hash *= 16777619u;
        value >>= 8;
    }
}

void xucg_fnv1a_feed_u64(uint32_t *hash, uint64_t value)
{
    for (int b = 0; b < 8; ++b) {
        *hash ^= static_cast<uint8_t>(value & 0xffu);
        *hash *= 16777619u;
        value >>= 8;
    }
}

/* 主机名 FNV-1a 哈希：同节点进程得到相同值（进程拓扑分组的依据） */
uint64_t xucg_host_hash()
{
    char host[256] = {0};
    if (::gethostname(host, sizeof(host) - 1) != 0)
        return 0;
    uint64_t h = 1469598103934665603ULL;
    for (const char *p = host; *p != '\0'; ++p) {
        h ^= static_cast<uint8_t>(*p);
        h *= 1099511628211ULL;
    }
    return h;
}

/* NUMA 节点探测：调用线程亲和性 CPU 集合 ∩ 各 NUMA 节点 cpulist，
 * 取交集最大的节点号；探测失败返回 -1（调用方据此全组一致降级） */
int32_t xucg_detect_numa()
{
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) != 0)
        return -1;
    DIR *d = opendir("/sys/devices/system/node");
    if (d == nullptr)
        return -1;
    int32_t best = -1, best_cnt = 0;
    struct dirent *e = nullptr;
    char path[96], buf[1024];
    while ((e = readdir(d)) != nullptr) {
        int n = -1;
        if (sscanf(e->d_name, "node%d", &n) != 1 || n < 0)
            continue;
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", n);
        FILE *f = std::fopen(path, "r");
        if (f == nullptr)
            continue;
        std::memset(buf, 0, sizeof(buf));
        const char *got = std::fgets(buf, sizeof(buf), f);
        std::fclose(f);
        if (got == nullptr)
            continue;
        int cnt = 0;
        for (char *tok = std::strtok(buf, ","); tok != nullptr; tok = std::strtok(nullptr, ",")) {
            int lo = -1, hi = -1;
            if (std::sscanf(tok, "%d-%d", &lo, &hi) == 2) {
                if (lo > hi)
                    continue;
                hi = std::min(hi, CPU_SETSIZE - 1);
                for (int c = std::max(lo, 0); c <= hi; ++c)
                    if (CPU_ISSET(c, &set))
                        ++cnt;
            } else if (std::sscanf(tok, "%d", &lo) == 1) {
                if (lo >= 0 && lo < CPU_SETSIZE && CPU_ISSET(lo, &set))
                    ++cnt;
            }
        }
        if (cnt > best_cnt) {
            best_cnt = cnt;
            best = n;
        }
    }
    closedir(d);
    return best_cnt > 0 ? best : -1;
}

/* 进程拓扑喂入开关（KUCCL_TOPOLOGY，默认 on）：
 * on  = 主机名哈希 OOB 交换喂 node_id（UCG 节点感知算法的输入）
 * off = 平铺模型（KUCCL_TOPOLOGY=0，A/B 对照 / 回退用） */
static bool topology_enabled()
{
    static const bool v = [] {
        const char *e = std::getenv("KUCCL_TOPOLOGY");
        return e == nullptr || e[0] == '\0' || std::strcmp(e, "0") != 0;
    }();
    return v;
}

/* ════════════════════════════════════════════════════════════
 * UCG 回调
 * ════════════════════════════════════════════════════════════*/

ucg_status_t xucg_ucg_oob_cb(const void *sendbuf, void *recvbuf, int32_t count, void *group)
{
    auto *xctx = static_cast<XucgComm *>(group);
    if (!xctx || !xctx->owner || !xctx->owner->has_oob)
        return UCG_ERR_INVALID_PARAM;
    int rc = detail::oob_exchange(&xctx->owner->oob_config, sendbuf, recvbuf, static_cast<int>(count));
    return (rc == 0) ? UCG_OK : UCG_ERR_INVALID_PARAM;
}

/* group_rank → global_rank 映射 */
bool xucg_group_rank_to_global(const XucgComm *xctx, int group_rank, int *global_rank)
{
    if (!xctx || !global_rank || group_rank < 0 || group_rank >= xctx->owner->nranks)
        return false;
    if (xctx->rank_map != nullptr && static_cast<uint32_t>(group_rank) < xctx->rank_map_size) {
        *global_rank = static_cast<int>(xctx->rank_map[group_rank]);
    } else {
        *global_rank = group_rank;
    }
    return true;
}

/* UCG 要建 endpoint 时回调：
 * group_rank → rank_map → 全局地址表 → ucp_ep_create */
void *xucg_get_ucp_ep(void * /*arg*/, void *oob_group, int group_rank)
{
    auto *xctx = static_cast<XucgComm *>(oob_group);
    const int rank = xctx ? xctx->owner->rank : -1;

    int global_peer = -1;
    if (!xucg_group_rank_to_global(xctx, group_rank, &global_peer)) {
        KuCCL_LOG_WARN(rank, "get_ucp_ep: invalid group_rank=%d", group_rank);
        return nullptr;
    }

    if (global_peer < 0 || global_peer >= g_xucg.global_nranks) {
        KuCCL_LOG_WARN(rank, "get_ucp_ep: global rank %d out of [0,%d)", global_peer, g_xucg.global_nranks);
        return nullptr;
    }

    if (!g_xucg.world_addr_lens || g_xucg.world_addr_lens[global_peer] == 0) {
        KuCCL_LOG_WARN(rank, "get_ucp_ep: no address for group_rank=%d global_rank=%d", group_rank, global_peer);
        return nullptr;
    }

    ucp_address_t *peer_addr =
        reinterpret_cast<ucp_address_t *>(g_xucg.world_addrs + static_cast<size_t>(global_peer) * g_xucg.max_addr_len);

    ucp_ep_params_t ep_params;
    std::memset(&ep_params, 0, sizeof(ep_params));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = peer_addr;

    ucp_ep_h ep = nullptr;
    const ucs_status_t st = ucp_ep_create(g_xucg.ucp_worker, &ep_params, &ep);
    if (st != UCS_OK) {
        KuCCL_LOG_WARN(rank, "ucp_ep_create FAILED: rank=%d status=%s", group_rank, ucs_status_string(st));
        return nullptr;
    }
    return static_cast<void *>(ep);
}

void *xucg_get_ucp_worker(void *)
{
    return static_cast<void *>(g_xucg.ucp_worker);
}
void *xucg_get_ucp_context(void *)
{
    return static_cast<void *>(g_xucg.ucp_context);
}

/* 进程拓扑查询：UCG 规划节点感知算法时回调（ctx 进程序 = 首通信域 rank 空间；
 * 子组经各 group 的 rank_map 映射回该空间查询）。
 * node_id 仅在拓扑交换成功（topology_ok，全组一致）时上报——UCG 的 locations[]
 * 用 malloc 分配、仅对上报者赋值，部分上报会被当数组下标用（越界写），
 * 因此必须全组同报或同不报。 */
ucg_status_t xucg_get_proc_info(ucg_rank_t rank, ucg_proc_info_t **proc)
{
    if (proc == nullptr)
        return UCG_ERR_INVALID_PARAM;
    *proc = static_cast<ucg_proc_info_t *>(std::malloc(sizeof(ucg_proc_info_t)));
    if (*proc == nullptr)
        return UCG_ERR_NO_MEMORY;
    std::memset(*proc, 0, sizeof(**proc));
    (*proc)->size = sizeof(ucg_proc_info_t);
    (*proc)->num_addr_desc = 0;
    g_topo_queries.fetch_add(1, std::memory_order_relaxed);
    if (g_xucg.topology_ok && rank >= 0 && static_cast<uint32_t>(rank) < g_xucg.node_id.size()) {
        g_topo_reported.fetch_add(1, std::memory_order_relaxed);
        (*proc)->location.field_mask = UCG_LOCATION_FIELD_NODE_ID;
        (*proc)->location.node_id = g_xucg.node_id[static_cast<uint32_t>(rank)];
        if (g_xucg.socket_ok && rank < static_cast<ucg_rank_t>(g_xucg.socket_id.size())) {
            (*proc)->location.field_mask |= UCG_LOCATION_FIELD_SOCKET_ID;
            (*proc)->location.socket_id = g_xucg.socket_id[static_cast<uint32_t>(rank)];
        }
    }
    return UCG_OK;
}

/* ════════════════════════════════════════════════════════════
 * 两阶段变长地址 allgather
 * ════════════════════════════════════════════════════════════*/

kucclResult_t xucg_build_world_addr_table(kucclComm *world_comm)
{
    const int rank = world_comm->rank, ws = world_comm->nranks;

    g_xucg.world_addr_lens = static_cast<size_t *>(std::calloc(ws, sizeof(size_t)));
    if (!g_xucg.world_addr_lens) {
        KuCCL_LOG_ERROR(rank, "calloc world lens FAILED");
        return KuCCL_FAIL;
    }

    /* 阶段 A：allgather 各自的 address_len */
    if (detail::oob_exchange(&world_comm->oob_config, &g_xucg.my_address_len, g_xucg.world_addr_lens,
                             static_cast<int>(sizeof(size_t))) != 0) {
        KuCCL_LOG_ERROR(rank, "OOB allgather (addr lens) FAILED");
        return KuCCL_FAIL;
    }
    g_xucg.max_addr_len = 0;
    for (int i = 0; i < ws; ++i)
        g_xucg.max_addr_len = std::max(g_xucg.max_addr_len, g_xucg.world_addr_lens[i]);

    /* 阶段 B：本地地址补零到 max_addr_len 后 allgather */
    g_xucg.world_addrs = static_cast<uint8_t *>(std::calloc(g_xucg.max_addr_len * ws, 1));
    if (!g_xucg.world_addrs) {
        KuCCL_LOG_ERROR(rank, "calloc world addrs FAILED");
        return KuCCL_FAIL;
    }
    {
        std::vector<uint8_t> padded(g_xucg.max_addr_len, 0u);
        std::memcpy(padded.data(), g_xucg.my_address, g_xucg.my_address_len);
        if (detail::oob_exchange(&world_comm->oob_config, padded.data(), g_xucg.world_addrs,
                                 static_cast<int>(g_xucg.max_addr_len)) != 0) {
            KuCCL_LOG_ERROR(rank, "OOB allgather (addrs) FAILED");
            return KuCCL_FAIL;
        }
    }
    KuCCL_LOG_INFO(rank, "world addr table built (N=%d, max_len=%zu)", ws, g_xucg.max_addr_len);
    return KuCCL_SUCCESS;
}

/* ════════════════════════════════════════════════════════════
 * 进程级一次性初始化
 * ════════════════════════════════════════════════════════════*/

kucclResult_t xucg_process_init_once(kucclComm *world_comm)
{
    if (g_xucg.initialized)
        return KuCCL_SUCCESS;
    const int rank = world_comm->rank, ws = world_comm->nranks;
    ucs_status_t st;

    /* 先挂后端私有状态（ucg_init 的 oob_group cookie 需要） */
    if (xucg_ensure_ctx(world_comm) == nullptr)
        return KuCCL_FAIL;

    /* (a) UCP context（xucg 后端私有资源层） */
    ucp_config_t *config = nullptr;
    st = ucp_config_read(nullptr, nullptr, &config);
    if (st != UCS_OK) {
        KuCCL_LOG_ERROR(rank, "ucp_config_read FAILED");
        return KuCCL_FAIL;
    }

    ucp_params_t up;
    std::memset(&up, 0, sizeof(up));
    up.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE | UCP_PARAM_FIELD_TAG_SENDER_MASK |
                    UCP_PARAM_FIELD_MT_WORKERS_SHARED | UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
    up.features = UCP_FEATURE_TAG;
    up.request_size = sizeof(kuccl_request_placeholder) + 4;
    up.tag_sender_mask = 0x800000fffffffffful;
    up.mt_workers_shared = 0;
    up.estimated_num_eps = static_cast<size_t>(ws);

    st = ucp_init(&up, config, &g_xucg.ucp_context);
    ucp_config_release(config);
    if (st != UCS_OK) {
        KuCCL_LOG_ERROR(rank, "ucp_init FAILED");
        return KuCCL_FAIL;
    }

    /* (b) UCP worker */
    ucp_worker_params_t wp;
    std::memset(&wp, 0, sizeof(wp));
    wp.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wp.thread_mode = UCS_THREAD_MODE_SINGLE;
    st = ucp_worker_create(g_xucg.ucp_context, &wp, &g_xucg.ucp_worker);
    if (st != UCS_OK) {
        KuCCL_LOG_ERROR(rank, "ucp_worker_create FAILED");
        return KuCCL_FAIL;
    }

    /* (c) UCP worker address */
    ucp_address_t *addr = nullptr;
    size_t alen = 0;
    st = ucp_worker_get_address(g_xucg.ucp_worker, &addr, &alen);
    if (st != UCS_OK) {
        KuCCL_LOG_ERROR(rank, "ucp_worker_get_address FAILED");
        return KuCCL_FAIL;
    }
    void *acopy = nullptr;
    if (posix_memalign(&acopy, 64, alen) != 0) {
        KuCCL_LOG_ERROR(rank, "posix_memalign FAILED");
        return KuCCL_FAIL;
    }
    std::memcpy(acopy, addr, alen);
    ucp_worker_release_address(g_xucg.ucp_worker, addr);
    g_xucg.my_address = static_cast<ucp_address_t *>(acopy);
    g_xucg.my_address_len = alen;

    g_xucg.global_rank = rank;
    g_xucg.global_nranks = ws;

    /* (d) 两阶段 OOB allgather 收集所有进程地址 */
    if (xucg_build_world_addr_table(world_comm) != KuCCL_SUCCESS)
        return KuCCL_FAIL;

    /* (d2) 进程拓扑（KUCCL_TOPOLOGY=0 可关，默认开）：主机名哈希 OOB 交换。
     * 不做任何"按 rank 推算节点"的假设（一机多作业 / 进程数不均 / 换 launcher
     * 都会使 rank/ppn 推算出错，猜错比不喂更糟——UCG 会按错误边界建组）；
     * LOCAL_WORLD_SIZE 只作交叉校验告警，不作数据源。
     * UCG 要求拓扑全组同报或同不报（其 locations[] 为 malloc 分配、仅对
     * 上报者赋值，部分上报的未初始化 node_id 会被当数组下标用导致越界写），
     * 故加一轮 1 字节全组表决：任一 rank 交换失败 → 全组统一降级平铺。 */
    g_xucg.topology_ok = false;
    if (topology_enabled()) {
        const uint64_t my_hash = xucg_host_hash();
        std::vector<uint64_t> hashes(static_cast<std::size_t>(ws));
        const bool hash_ok = detail::oob_exchange(&world_comm->oob_config, &my_hash, hashes.data(),
                                                  static_cast<int>(sizeof(uint64_t))) == 0;

        /* NUMA（socket）交换：与 node_id 同源的全组一致原则——任一 rank
         * 探测失败（发 -1）→ 全组不喂 socket（UCG 的 socket 数组仅对
         * 上报者赋值，部分上报 = 未初始化值被当数组下标用）。
         * 探测失败不影响 node_id 的喂入；NUMA 交换失败计入表决。 */
        std::vector<int32_t> numa(static_cast<std::size_t>(ws), -1);
        const int32_t my_numa = xucg_detect_numa();
        const bool numa_xchg_ok = detail::oob_exchange(&world_comm->oob_config, &my_numa, numa.data(),
                                                       static_cast<int>(sizeof(int32_t))) == 0;
        bool numa_ok_all = numa_xchg_ok;
        for (int i = 0; numa_ok_all && i < ws; ++i)
            numa_ok_all = numa[static_cast<std::size_t>(i)] >= 0;

        /* 表决轮：哈希交换与 NUMA 交换都成功的 rank 投 1 → 全组一致决策 */
        const uint8_t my_vote = (hash_ok && numa_xchg_ok) ? 1 : 0;
        std::vector<uint8_t> votes(static_cast<std::size_t>(ws));
        const bool vote_ok = detail::oob_exchange(&world_comm->oob_config, &my_vote, votes.data(), 1) == 0;
        bool all_ok = (hash_ok && numa_xchg_ok) && vote_ok;
        for (int i = 0; all_ok && i < ws; ++i)
            all_ok = votes[static_cast<std::size_t>(i)] != 0;

        if (!all_ok) {
            KuCCL_LOG_ERROR(rank, "topology exchange/vote FAILED — degrade to flat model "
                                  "(group-consistent)");
        } else {
            g_xucg.host_hash = std::move(hashes);
            g_xucg.node_id.assign(static_cast<std::size_t>(ws), 0);
            /* 按"首次出现的顺序"给每个主机名编一个 node_id */
            std::map<uint64_t, int32_t> first_seen;
            int32_t next_node = 0;
            for (int i = 0; i < ws; ++i) {
                auto ins = first_seen.emplace(g_xucg.host_hash[static_cast<std::size_t>(i)], next_node);
                g_xucg.node_id[static_cast<std::size_t>(i)] = ins.first->second;
                if (ins.second)
                    ++next_node;
            }
            g_xucg.num_local_procs = 0;
            for (int i = 0; i < ws; ++i)
                if (g_xucg.host_hash[static_cast<std::size_t>(i)] == my_hash)
                    ++g_xucg.num_local_procs;
            g_xucg.topology_ok = true;
            g_xucg.nnode = static_cast<int32_t>(first_seen.size());
            if (numa_ok_all) {
                /* socket_id：NUMA 节点号按首次出现顺序稠密化 */
                g_xucg.socket_id.assign(static_cast<std::size_t>(ws), 0);
                std::map<int32_t, int32_t> first_socket;
                int32_t next_socket = 0;
                for (int i = 0; i < ws; ++i) {
                    auto ins = first_socket.emplace(numa[static_cast<std::size_t>(i)], next_socket);
                    g_xucg.socket_id[static_cast<std::size_t>(i)] = ins.first->second;
                    if (ins.second)
                        ++next_socket;
                }
                g_xucg.nsocket = next_socket;
                g_xucg.socket_ok = true;
            }
            if (g_xucg.socket_ok) {
                KuCCL_LOG_INFO(rank,
                               "topology(src=host-hash): nodes=%d ppn=%u sockets=%d "
                               "my_node=%d my_socket=%d",
                               static_cast<int>(first_seen.size()), g_xucg.num_local_procs, g_xucg.nsocket,
                               g_xucg.node_id[static_cast<std::size_t>(rank)],
                               g_xucg.socket_id[static_cast<std::size_t>(rank)]);
            } else {
                KuCCL_LOG_INFO(rank,
                               "topology(src=host-hash): nodes=%d ppn=%u my_node=%d "
                               "(socket not fed)",
                               static_cast<int>(first_seen.size()), g_xucg.num_local_procs,
                               g_xucg.node_id[static_cast<std::size_t>(rank)]);
            }
            /* 交叉校验（仅告警不作数据源）：torchrun 注入值与实测不一致
             * 说明 hostname 碰撞或一机多作业——拓扑仍以实测为准 */
            if (const char *lws = std::getenv("LOCAL_WORLD_SIZE")) {
                const long hint = std::atol(lws);
                if (hint > 0 && static_cast<unsigned long>(hint) != g_xucg.num_local_procs)
                    KuCCL_LOG_ERROR(rank,
                                    "topology MISMATCH: LOCAL_WORLD_SIZE=%ld but "
                                    "host-hash ppn=%u (hash collision or multi-job "
                                    "per host?)",
                                    hint, g_xucg.num_local_procs);
            }
        }
    }

    /* (e) UCG global init — 注入 UCP 资源回调（ucg 编程模型） */
    ucg_global_params_t gp;
    std::memset(&gp, 0, sizeof(gp));
    gp.field_mask = UCG_GLOBAL_PARAMS_FIELD_OOB_RESOURCE;
    gp.oob_resource.get_ucp_ep = xucg_get_ucp_ep;
    gp.oob_resource.get_ucp_worker = xucg_get_ucp_worker;
    gp.oob_resource.get_ucp_context = xucg_get_ucp_context;
    gp.oob_resource.arg = nullptr;
    if (ucg_global_init(&gp) != UCG_OK) {
        KuCCL_LOG_ERROR(rank, "ucg_global_init FAILED");
        return KuCCL_FAIL;
    }

    /* (f) UCG init。oob_group.num_local_procs 全树唯一消费点是 UCX 的
     * estimated_num_ppn 性能提示（planc_ucx_context.c），不参与算法选择——
     * 算法/策略表分档只看 get_proc_info 上报的 node_id。 */
    ucg_params_t cp;
    std::memset(&cp, 0, sizeof(cp));
    cp.field_mask = UCG_PARAMS_FIELD_OOB_GROUP | UCG_PARAMS_FIELD_THREAD_MODE | UCG_PARAMS_FIELD_PROC_INFO_CB;
    cp.oob_group.allgather = xucg_ucg_oob_cb;
    cp.oob_group.myrank = static_cast<ucg_rank_t>(rank);
    cp.oob_group.size = static_cast<uint32_t>(ws);
    cp.oob_group.group = world_comm->be_ctx;
    cp.oob_group.num_local_procs = g_xucg.topology_ok ? g_xucg.num_local_procs : 0;
    cp.thread_mode = UCG_THREAD_MODE_SINGLE;
    cp.get_proc_info = xucg_get_proc_info;

    ucg_config_h ucfg = nullptr;
    if (ucg_config_read(NULL, NULL, &ucfg) != UCG_OK) {
        KuCCL_LOG_ERROR(rank, "ucg_config_read FAILED");
        return KuCCL_FAIL;
    }
    ucg_status_t us = ucg_init(&cp, ucfg, &g_xucg.ucg_context);
    ucg_config_release(ucfg);
    if (us != UCG_OK) {
        KuCCL_LOG_ERROR(rank, "ucg_init FAILED st=%d", static_cast<int>(us));
        return KuCCL_FAIL;
    }

    g_xucg.initialized = true;
    KuCCL_LOG_INFO(rank, "process-level UCX/UCG init done (once)");
    return KuCCL_SUCCESS;
}

/* ════════════════════════════════════════════════════════════
 * dt/op 缓存：创建一次终身复用
 * ════════════════════════════════════════════════════════════*/

constexpr int XUCG_DT_CACHE_SIZE = 8; /* kucclDataType_t 取值数（0..7） */
constexpr int XUCG_OP_CACHE_SIZE = 3; /* kucclRedOp_t 取值数（0..2） */
ucg_dt_h g_dt_cache[XUCG_DT_CACHE_SIZE] = {};
ucg_op_h g_op_cache[XUCG_OP_CACHE_SIZE] = {};

/* 集合接口的 dtype 语义开关（KUCCL_VDT）：
 * bytes（默认）= UINT8 + 字节数；elem = 真实 dtype + 元素数。
 * 两种语义下 UCG 的 plan 选择不同。
 * allreduce 不受此开关影响：归约必须真实 dtype。 */
static bool vdt_bytes()
{
    static const bool v = [] {
        const char *e = std::getenv("KUCCL_VDT");
        return e == nullptr || e[0] == '\0' || std::strcmp(e, "bytes") == 0;
    }();
    return v;
}

static size_t kuccl_dt_bytes(kucclDataType_t datatype)
{
    switch (datatype) {
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
        default:
            return 0;
    }
}

kucclResult_t get_cached_dt(kucclDataType_t datatype, ucg_dt_h *out_dt)
{
    const int idx = static_cast<int>(datatype);
    if (idx < 0 || idx >= XUCG_DT_CACHE_SIZE)
        return KuCCL_FAIL;
    if (g_dt_cache[idx] != nullptr) {
        *out_dt = g_dt_cache[idx];
        return KuCCL_SUCCESS;
    }

    ucg_dt_params_t dt_params;
    std::memset(&dt_params, 0, sizeof(dt_params));
    decltype(dt_params.type) ucg_dtype;
    size_t dt_size;
    switch (datatype) {
        case kucclInt8:
            ucg_dtype = UCG_DT_TYPE_INT8;
            dt_size = 1;
            break;
        case kucclUint8:
            ucg_dtype = UCG_DT_TYPE_UINT8;
            dt_size = 1;
            break;
        case kucclInt32:
            ucg_dtype = UCG_DT_TYPE_INT32;
            dt_size = 4;
            break;
        case kucclUint32:
            ucg_dtype = UCG_DT_TYPE_UINT32;
            dt_size = 4;
            break;
        case kucclInt64:
            ucg_dtype = UCG_DT_TYPE_INT64;
            dt_size = 8;
            break;
        case kucclUint64:
            ucg_dtype = UCG_DT_TYPE_UINT64;
            dt_size = 8;
            break;
        case kucclFloat:
            ucg_dtype = UCG_DT_TYPE_FP32;
            dt_size = 4;
            break;
        case kucclDouble:
            ucg_dtype = UCG_DT_TYPE_FP64;
            dt_size = 8;
            break;
        default:
            return KuCCL_FAIL;
    }
    dt_params.field_mask = UCG_DT_PARAMS_FIELD_TYPE | UCG_DT_PARAMS_FIELD_SIZE;
    dt_params.type = ucg_dtype;
    dt_params.size = dt_size;

    if (ucg_dt_create(&dt_params, &g_dt_cache[idx]) != UCG_OK)
        return KuCCL_FAIL;
    *out_dt = g_dt_cache[idx];
    return KuCCL_SUCCESS;
}

kucclResult_t get_cached_op(kucclRedOp_t reduction_op, ucg_op_h *out_op)
{
    const int idx = static_cast<int>(reduction_op);
    if (idx < 0 || idx >= XUCG_OP_CACHE_SIZE)
        return KuCCL_FAIL;
    if (g_op_cache[idx] != nullptr) {
        *out_op = g_op_cache[idx];
        return KuCCL_SUCCESS;
    }

    ucg_op_params_t op_params;
    std::memset(&op_params, 0, sizeof(op_params));
    decltype(op_params.type) ucg_optype;
    switch (reduction_op) {
        case kucclSum:
            ucg_optype = UCG_OP_TYPE_SUM;
            break;
        case kucclMin:
            ucg_optype = UCG_OP_TYPE_MIN;
            break;
        case kucclMax:
            ucg_optype = UCG_OP_TYPE_MAX;
            break;
        default:
            return KuCCL_FAIL;
    }
    op_params.field_mask = UCG_OP_PARAMS_FIELD_TYPE;
    op_params.type = ucg_optype;

    if (ucg_op_create(&op_params, &g_op_cache[idx]) != UCG_OK)
        return KuCCL_FAIL;
    *out_op = g_op_cache[idx];
    return KuCCL_SUCCESS;
}

/* datatype 的元素字节大小（不创建 UCG dt，零开销） */
size_t xucg_dt_size(kucclDataType_t dt)
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
    return 0;
}

/* ════════════════════════════════════════════════════════════
 * KUCCL_PROFILE —— 热路径剖析（KUCCL_PROFILE=1 开启，默认关零开销）
 *
 * 每算子累计：调用数 / init 耗时 / start 耗时 / 端到端总耗时 / test 轮询次数，
 * 进程退出时打印汇总表。剖析的是 UCG 调用面耗时（后端内聚）。
 * ==========================================================================*/

constexpr int XUCG_PROF_NOPS = 4; /* 0..3：ag/ar/bcast/a2a */
const char *const XUCG_PROF_NAMES[XUCG_PROF_NOPS] = {"allgather", "allreduce", "broadcast", "alltoall"};

struct ProfOpStats {
    uint64_t n = 0;
    uint64_t test_calls = 0;
    double init_us = 0.0;
    double start_us = 0.0;
    double total_us = 0.0; /* 启动→kucclTest 判定完成 */
};

/* ★ 声明必须在 prof_enabled() 之前：其 lambda 里的名字查找在定义点进行 */
ProfOpStats g_prof_stats[XUCG_PROF_NOPS];
int g_prof_rank = -1;

inline bool prof_enabled()
{
    static const bool v = [] {
        const char *e = std::getenv("KUCCL_PROFILE");
        if (e == nullptr || e[0] == '\0' || std::strcmp(e, "0") == 0)
            return false;
        std::atexit([] { /* dump 一次：只打全局 rank 0（集合通信锁步，各 rank 统计近似；
                    128 份同表只刷屏。要看别的 rank 临时改此条件） */
                         if (g_prof_rank != 0)
                             return;
                         std::fprintf(stdout, "[KUCCL_PROFILE][rank %d] %-10s %10s %12s %12s %12s %14s\n", g_prof_rank,
                                      "op", "n", "init_us/op", "start_us/op", "total_us/op", "test/op");
                         for (int i = 0; i < XUCG_PROF_NOPS; ++i) {
                             const ProfOpStats &s = g_prof_stats[i];
                             if (s.n == 0)
                                 continue;
                             std::fprintf(stdout, "[KUCCL_PROFILE][rank %d] %-10s %10llu %12.2f %12.2f %12.2f %14.1f\n",
                                          g_prof_rank, XUCG_PROF_NAMES[i], static_cast<unsigned long long>(s.n),
                                          s.init_us / s.n, s.start_us / s.n, s.total_us / s.n,
                                          static_cast<double>(s.test_calls) / s.n);
                         }
                         std::fflush(stdout);
        });
        return true;
    }();
    return v;
}

inline double prof_us_since(const std::chrono::steady_clock::time_point &t0)
{
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

/* ════════════════════════════════════════════════════════════
 * KUCCL_PERSISTENT_REQ —— 持久化请求（默认关）
 *
 * ★ plan 一次构建终身复用，热路径零 init；ucg.h 明确支持
 *   request 完成后再次 start。
 *
 * 机制：同 key（算子/dtype/count/缓冲指针/root/comm）的集合调用，
 *   完成后不销毁 ucg_request，下次直接 ucg_request_start 复用。
 *
 * AllGather/AllToAll 的 counts/displs 随单次请求释放，
 * 请求入池复用后指针失效。故仅 AllReduce / Broadcast 入池。
 * ════════════════════════════════════════════════════════════*/

enum {
    XUCG_PERSIST_AR = 1,
    XUCG_PERSIST_BCAST = 2
};

struct PersistentEntry {
    /* key */
    uint32_t op;
    kucclDataType_t dt;
    uint64_t count;
    const void *send;
    void *recv;
    int root;
    kucclComm *comm;
    /* value */
    ucg_request_h req = nullptr;
    bool in_flight = false;
    bool valid = false;
};

constexpr int XUCG_PERSIST_MAX = 32;
PersistentEntry g_persist[XUCG_PERSIST_MAX];

inline bool persist_enabled()
{
    static const bool v = [] {
        const char *e = std::getenv("KUCCL_PERSISTENT_REQ");
        return e != nullptr && e[0] != '\0' && std::strcmp(e, "0") != 0;
    }();
    return v;
}

/* 找可复用 entry：key 全匹配且不在飞行中 */
inline PersistentEntry *persist_find(uint32_t op, kucclDataType_t dt, uint64_t count, const void *send, void *recv,
                                     int root, kucclComm *comm)
{
    for (int i = 0; i < XUCG_PERSIST_MAX; ++i) {
        PersistentEntry &e = g_persist[i];
        if (e.valid && !e.in_flight && e.op == op && e.dt == dt && e.count == count && e.send == send &&
            e.recv == recv && e.root == root && e.comm == comm)
            return &e;
    }
    return nullptr;
}

/* 找空位存新 entry（无空位/重复则放弃入池，不影响正确性） */
inline PersistentEntry *persist_store_slot(uint32_t op, kucclDataType_t dt, uint64_t count, const void *send,
                                           void *recv, int root, kucclComm *comm)
{
    bool dup = false;
    int free_idx = -1;
    for (int i = 0; i < XUCG_PERSIST_MAX; ++i) {
        PersistentEntry &e = g_persist[i];
        if (e.valid && e.op == op && e.dt == dt && e.count == count && e.send == send && e.recv == recv &&
            e.root == root && e.comm == comm) {
            dup = true;
            break;
        }
        if (!e.valid && free_idx < 0)
            free_idx = i;
    }
    if (dup || free_idx < 0)
        return nullptr;
    return &g_persist[free_idx];
}

/* comm 销毁/abort 时清池（调用方已持锁）。
 * 销毁时仍有 in_flight 请求 = 调用方使用错误，这里尽力 cleanup。 */
void purge_persistent_for_comm(kucclComm_t comm)
{
    for (int i = 0; i < XUCG_PERSIST_MAX; ++i) {
        PersistentEntry &e = g_persist[i];
        if (e.valid && e.comm == comm) {
            if (e.req)
                ucg_request_cleanup(e.req);
            e = PersistentEntry{}; /* 归零（valid=false） */
        }
    }
}

/* request_info 公共填充（HOST 内存，所有算子一致） */
inline ucg_request_info_t make_request_info()
{
    ucg_request_info_t info;
    std::memset(&info, 0, sizeof(info));
    info.field_mask = UCG_REQUEST_INFO_FIELD_MEM_TYPE;
    info.mem_type = UCG_MEM_TYPE_HOST;
    return info;
}

} /* anonymous namespace */

/* ════════════════════════════════════════════════════════════
 * 集合启动：一次调用的 (缓存)dt/op → request_init → start
 * ════════════════════════════════════════════════════════════*/

static kucclResult_t xucg_coll_start(kucclComm *comm, CollOp op, const CollParams *p, kucclRequest *req)
{
    const int rank = comm->rank;
    const bool prof = prof_enabled();
    if (prof && g_prof_rank < 0)
        g_prof_rank = rank;
    const auto prof_t_setup = std::chrono::steady_clock::now();

    auto *xctx = static_cast<XucgComm *>(comm->be_ctx);
    auto *priv = new (std::nothrow) XucgReq();
    if (priv == nullptr) {
        detail::err_record("xucg: alloc request state FAILED");
        return KuCCL_FAIL;
    }
    priv->op_id = static_cast<uint32_t>(op);

    /* ── 持久化快路径：同 key 集合直接 start，零 init（默认关）── */
    if (persist_enabled() && (op == CollOp::AllReduce || op == CollOp::Broadcast)) {
        const uint32_t pop = (op == CollOp::AllReduce) ? XUCG_PERSIST_AR : XUCG_PERSIST_BCAST;
        PersistentEntry *e =
            persist_find(pop, p->datatype, static_cast<uint64_t>(p->count), p->sendbuff, p->recvbuff, p->root, comm);
        if (e != nullptr) {
            if (ucg_request_start(e->req) == UCG_OK) {
                e->in_flight = true;
                priv->ucg_req = e->req;
                priv->persist_entry = e;
                req->priv = priv;
                if (prof) {
                    req->t0 = prof_t_setup;
                    g_prof_stats[priv->op_id].n++;
                    g_prof_stats[priv->op_id].start_us += prof_us_since(prof_t_setup);
                }
                return KuCCL_SUCCESS;
            }
            /* restart 失败 → UCG 不支持复用，清掉 entry 走全新路径 */
            ucg_request_cleanup(e->req);
            *e = PersistentEntry{};
        }
    }

    /* 1. dt / op（进程级缓存）。allreduce 用真实 dtype + 元素数（归约必需）；
     * bcast/allgatherv/alltoallv 由 KUCCL_VDT 决定，见 vdt_bytes 注释。 */
    ucg_dt_h dt = nullptr;
    ucg_op_h op_h = nullptr;
    if (get_cached_dt(p->datatype, &dt) != KuCCL_SUCCESS) {
        KuCCL_LOG_ERROR(rank, "op=%d: get_cached_dt FAILED", static_cast<int>(op));
        delete priv;
        return KuCCL_FAIL;
    }
    if (op == CollOp::AllReduce && get_cached_op(p->redop, &op_h) != KuCCL_SUCCESS) {
        KuCCL_LOG_ERROR(rank, "allreduce: get_cached_op FAILED");
        delete priv;
        return KuCCL_FAIL;
    }

    /* 2. request_init（按算子分发到对应 UCG 公有接口）
     * info 挂 priv 保活：UCG init 之后仍可能读取 */
    priv->keep_info = make_request_info();
    ucg_request_info_t &info = priv->keep_info;
    const auto prof_t_init = std::chrono::steady_clock::now();
    ucg_request_h ucg_req = nullptr;
    ucg_status_t st = UCG_OK;

    switch (op) {
        case CollOp::AllGather: {
            /* 等长 allgatherv 参数（挂 priv 保活）。dtype 语义由 KUCCL_VDT 决定。 */
            const int nranks = comm->nranks;
            std::vector<int32_t> &recvcounts = priv->keep_v[0];
            std::vector<int32_t> &displs = priv->keep_v[1];
            ucg_dt_h vdt = dt;
            int32_t unit = static_cast<int32_t>(p->count);
            if (vdt_bytes()) {
                const size_t nb = static_cast<size_t>(p->count) * kuccl_dt_bytes(p->datatype);
                if (nb > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                    detail::err_record("allgatherv bytes mode: per-rank bytes overflow int32");
                    delete priv;
                    return KuCCL_FAIL;
                }
                if (get_cached_dt(kucclUint8, &vdt) != KuCCL_SUCCESS) {
                    detail::err_record("allgatherv: get uint8 dt FAILED");
                    delete priv;
                    return KuCCL_FAIL;
                }
                unit = static_cast<int32_t>(nb);
            }
            recvcounts.assign(nranks, unit);
            displs.assign(nranks, 0);
            for (int i = 1; i < nranks; ++i)
                displs[i] = displs[i - 1] + recvcounts[i - 1];
            st = ucg_request_allgatherv_init(p->sendbuff, unit, vdt, p->recvbuff, recvcounts.data(), displs.data(), vdt,
                                             xctx->group, &info, UCG_REQUEST_BLOCKING, &ucg_req);
            break;
        }
        case CollOp::AllReduce:
            /* count 是元素个数，不是字节数！ */
            st = ucg_request_allreduce_init(p->sendbuff, p->recvbuff, static_cast<int>(p->count), dt, op_h, xctx->group,
                                            &info, UCG_REQUEST_BLOCKING, &ucg_req);
            break;
        case CollOp::Broadcast: {
            /* dtype 语义随 KUCCL_VDT。原地：buf 既收也发。 */
            ucg_dt_h vdt = dt;
            int32_t unit = static_cast<int32_t>(p->count);
            if (vdt_bytes()) {
                const size_t nb = static_cast<size_t>(p->count) * kuccl_dt_bytes(p->datatype);
                if (nb > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                    detail::err_record("bcast bytes mode: bytes overflow int32");
                    delete priv;
                    return KuCCL_FAIL;
                }
                if (get_cached_dt(kucclUint8, &vdt) != KuCCL_SUCCESS) {
                    detail::err_record("bcast: get uint8 dt FAILED");
                    delete priv;
                    return KuCCL_FAIL;
                }
                unit = static_cast<int32_t>(nb);
            }
            st = ucg_request_bcast_init(p->recvbuff, unit, vdt, p->root, xctx->group, &info, UCG_REQUEST_BLOCKING,
                                        &ucg_req);
            break;
        }
        case CollOp::AllToAll: {
            /* 等长 alltoallv 参数（挂 priv 保活）。dtype 语义由 KUCCL_VDT 决定。 */
            const int nranks = comm->nranks;
            std::vector<int32_t> &sendcounts = priv->keep_v[0];
            std::vector<int32_t> &sdispls = priv->keep_v[1];
            std::vector<int32_t> &recvcounts = priv->keep_v[2];
            std::vector<int32_t> &rdispls = priv->keep_v[3];
            ucg_dt_h vdt = dt;
            int32_t unit = static_cast<int32_t>(p->count);
            if (vdt_bytes()) {
                const size_t nb = static_cast<size_t>(p->count) * kuccl_dt_bytes(p->datatype);
                if (nb > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                    detail::err_record("alltoallv bytes mode: per-peer bytes overflow int32");
                    delete priv;
                    return KuCCL_FAIL;
                }
                if (get_cached_dt(kucclUint8, &vdt) != KuCCL_SUCCESS) {
                    detail::err_record("alltoallv: get uint8 dt FAILED");
                    delete priv;
                    return KuCCL_FAIL;
                }
                unit = static_cast<int32_t>(nb);
            }
            sendcounts.assign(nranks, unit);
            sdispls.assign(nranks, 0);
            recvcounts.assign(nranks, unit);
            rdispls.assign(nranks, 0);
            for (int i = 1; i < nranks; ++i) {
                sdispls[i] = sdispls[i - 1] + sendcounts[i - 1];
                rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
            }
            st = ucg_request_alltoallv_init(p->sendbuff, sendcounts.data(), sdispls.data(), vdt, p->recvbuff,
                                            recvcounts.data(), rdispls.data(), vdt, xctx->group, &info,
                                            UCG_REQUEST_BLOCKING, &ucg_req);
            break;
        }
        default:
            detail::err_record("xucg: unsupported coll op %d", static_cast<int>(op));
            delete priv;
            return KuCCL_FAIL;
    }

    if (prof)
        g_prof_stats[priv->op_id].init_us += prof_us_since(prof_t_init);
    if (st != UCG_OK) {
        detail::err_record("xucg: ucg_request init FAILED op=%d st=%d", static_cast<int>(op), static_cast<int>(st));
        KuCCL_LOG_ERROR(rank, "op=%d: ucg_request init FAILED st=%d", static_cast<int>(op), static_cast<int>(st));
        delete priv;
        return KuCCL_FAIL;
    }

    /* 3. start */
    const auto prof_t_start = std::chrono::steady_clock::now();
    st = ucg_request_start(ucg_req);
    if (prof)
        g_prof_stats[priv->op_id].start_us += prof_us_since(prof_t_start);
    if (st != UCG_OK) {
        detail::err_record("xucg: ucg_request_start FAILED op=%d st=%d", static_cast<int>(op), static_cast<int>(st));
        KuCCL_LOG_ERROR(rank, "op=%d: ucg_request_start FAILED st=%d", static_cast<int>(op), static_cast<int>(st));
        ucg_request_cleanup(ucg_req);
        delete priv;
        return KuCCL_FAIL;
    }

    priv->ucg_req = ucg_req;
    req->priv = priv;

    /* 入池：下次同 key 调用免 init（默认关；池满/重复则放弃） */
    if (persist_enabled() && (op == CollOp::AllReduce || op == CollOp::Broadcast)) {
        const uint32_t pop = (op == CollOp::AllReduce) ? XUCG_PERSIST_AR : XUCG_PERSIST_BCAST;
        PersistentEntry *slot = persist_store_slot(pop, p->datatype, static_cast<uint64_t>(p->count), p->sendbuff,
                                                   p->recvbuff, p->root, comm);
        if (slot != nullptr) {
            slot->op = pop;
            slot->dt = p->datatype;
            slot->count = static_cast<uint64_t>(p->count);
            slot->send = p->sendbuff;
            slot->recv = p->recvbuff;
            slot->root = p->root;
            slot->comm = comm;
            slot->req = ucg_req;
            slot->in_flight = true;
            slot->valid = true;
            priv->persist_entry = slot;
        }
    }

    if (prof) {
        req->t0 = prof_t_setup;
        g_prof_stats[priv->op_id].n++;
    }
    return KuCCL_SUCCESS;
}

/* ════════════════════════════════════════════════════════════
 * 请求推进：ucg_request_test → 未就绪则 ucg_progress（UCG 公有推进）
 * 完成时回收 priv（持久化请求归还池）
 * ════════════════════════════════════════════════════════════*/

static kucclResult_t xucg_req_test(kucclRequest *req, int *flag, kucclStatus_t *status)
{
    auto *priv = static_cast<XucgReq *>(req->priv);
    const bool prof = prof_enabled();
    if (prof)
        priv->test_calls++;

    /* 先查状态（大概率已就绪，避免无效 progress） */
    ucg_status_t st = ucg_request_test(priv->ucg_req);

    /* 没就绪才推进（UCG 公有接口；worker 泵由后端私有资源层兜底） */
    if (st == UCG_INPROGRESS) {
        ucg_progress(g_xucg.ucg_context);
        ucp_worker_progress(g_xucg.ucp_worker); /* 后端私有的资源泵 */
        *flag = 0;
        *status = KuCCL_STATUS_PENDING;
        return KuCCL_SUCCESS;
    }

    /* 完成/出错时统一收尾：剖析累计 + 持久化归还/清理 */
    if (prof) {
        ProfOpStats &s = g_prof_stats[priv->op_id % XUCG_PROF_NOPS];
        s.total_us += prof_us_since(req->t0);
        s.test_calls += priv->test_calls;
    }
    auto *entry = static_cast<PersistentEntry *>(priv->persist_entry);

    if (st == UCG_OK) {
        *flag = 1;
        *status = KuCCL_STATUS_COMPLETE;
        if (entry != nullptr) {
            entry->in_flight = false; /* 归还池（ucg_req 保留） */
        } else {
            ucg_request_cleanup(priv->ucg_req);
        }
    } else {
        *flag = 1;
        *status = KuCCL_STATUS_ERROR;
        ucg_request_cleanup(priv->ucg_req);
        if (entry != nullptr)
            *entry = PersistentEntry{};
    }
    delete priv;
    req->priv = nullptr;
    return KuCCL_SUCCESS;
}

/* 取消请求（看门狗超时路径）：UCG cancel 语义——cleanup 在飞请求、
 * 持久化 entry 失效、回收私有状态 */
static void xucg_req_cancel(kucclRequest *req)
{
    auto *priv = static_cast<XucgReq *>(req->priv);
    if (priv == nullptr)
        return;
    if (priv->ucg_req != nullptr)
        ucg_request_cleanup(priv->ucg_req);
    auto *entry = static_cast<PersistentEntry *>(priv->persist_entry);
    if (entry != nullptr)
        *entry = PersistentEntry{};
    delete priv;
    req->priv = nullptr;
}

/* ════════════════════════════════════════════════════════════
 * 通信域生命周期：process_init / group_init / group_fini / flush
 * ════════════════════════════════════════════════════════════*/

static kucclResult_t xucg_process_init(kucclComm *comm)
{
    return xucg_process_init_once(comm);
}

static void xucg_process_finalize()
{
    if (!g_xucg.initialized)
        return;
    if (g_xucg.ucg_context)
        ucg_cleanup(g_xucg.ucg_context);
    std::free(g_xucg.world_addrs);
    std::free(g_xucg.world_addr_lens);
    if (g_xucg.my_address)
        std::free(g_xucg.my_address);
    if (g_xucg.ucp_worker)
        ucp_worker_destroy(g_xucg.ucp_worker);
    if (g_xucg.ucp_context)
        ucp_cleanup(g_xucg.ucp_context);
    g_xucg = XucgGlobal{};
}

static kucclResult_t xucg_group_init(kucclComm *comm)
{
    const int rank = comm->rank;
    const int nranks = comm->nranks;

    /* 复用（或创建）后端私有状态 */
    auto *xctx = xucg_ensure_ctx(comm);
    if (xctx == nullptr)
        return KuCCL_FAIL;

    /* 进程级一次性初始化（首次调用才真正执行；cookie 已就位） */
    if (xucg_process_init_once(comm) != KuCCL_SUCCESS) {
        return KuCCL_FAIL;
    }

    /* ── bootstrap：OOB 交换 global_rank + instance_token ── */
    std::vector<XucgBootstrapRecord> bootstrap_records(static_cast<std::size_t>(nranks));
    XucgBootstrapRecord my_record{};
    my_record.global_rank = static_cast<int32_t>(g_xucg.global_rank);
    my_record.instance_token = xucg_mix64(static_cast<uint64_t>(::getpid()) * 0x9e3779b97f4a7c15ULL +
                                          std::chrono::steady_clock::now().time_since_epoch().count());
    my_record.host_hash = xucg_host_hash();

    if (detail::oob_exchange(&comm->oob_config, &my_record, bootstrap_records.data(),
                             static_cast<int>(sizeof(XucgBootstrapRecord))) != 0) {
        KuCCL_LOG_ERROR(rank, "OOB bootstrap exchange FAILED");
        return KuCCL_FAIL; /* ctx 由 group_fini 统一回收 */
    }

    std::vector<int32_t> gids(static_cast<std::size_t>(nranks), -1);
    std::vector<uint64_t> tokens(static_cast<std::size_t>(nranks), 0);
    for (int i = 0; i < nranks; ++i) {
        const auto &record = bootstrap_records[static_cast<std::size_t>(i)];
        if (record.global_rank < 0 || record.global_rank >= g_xucg.global_nranks) {
            KuCCL_LOG_ERROR(rank, "invalid bootstrap global rank: peer=%d gr=%d", i,
                            static_cast<int>(record.global_rank));
            return KuCCL_FAIL; /* ctx 由 group_fini 统一回收 */
        }
        gids[i] = record.global_rank;
        tokens[i] = record.instance_token;
    }

    /* rank_map：local → global */
    xctx->rank_map = new (std::nothrow) ucg_rank_t[static_cast<std::size_t>(nranks)];
    if (!xctx->rank_map) {
        return KuCCL_FAIL; /* ctx 由 group_fini 统一回收 */
    }
    xctx->rank_map_size = static_cast<uint32_t>(nranks);
    for (int i = 0; i < nranks; ++i)
        xctx->rank_map[i] = static_cast<ucg_rank_t>(gids[static_cast<std::size_t>(i)]);

    /* 自检：rank_map 必须映射回本进程 */
    if (static_cast<int>(xctx->rank_map[rank]) != g_xucg.global_rank) {
        KuCCL_LOG_ERROR(rank, "RANK MAP SELF MISMATCH: local=%d mapped=%d actual=%d", rank,
                        static_cast<int>(xctx->rank_map[rank]), g_xucg.global_rank);
        return KuCCL_FAIL; /* rank_map/ctx 由 group_fini 统一回收 */
    }

    /* FNV-1a group_id 哈希 */
    uint32_t h = 2166136261u;
    xucg_fnv1a_feed_u32(&h, static_cast<uint32_t>(nranks));
    for (int i = 0; i < nranks; ++i)
        xucg_fnv1a_feed_u32(&h, static_cast<uint32_t>(gids[i]));
    for (int i = 0; i < nranks; ++i)
        xucg_fnv1a_feed_u64(&h, tokens[i]);
    xctx->group_id = h % 262144u;

    /* ucg_group_create */
    ucg_group_params_t gpm;
    std::memset(&gpm, 0, sizeof(gpm));
    gpm.field_mask = UCG_GROUP_PARAMS_FIELD_ID | UCG_GROUP_PARAMS_FIELD_SIZE | UCG_GROUP_PARAMS_FIELD_MYRANK |
                     UCG_GROUP_PARAMS_FIELD_RANK_MAP | UCG_GROUP_PARAMS_FIELD_OOB_GROUP;
    gpm.id = xctx->group_id;
    gpm.size = static_cast<uint32_t>(nranks);
    gpm.myrank = static_cast<ucg_rank_t>(rank);
    gpm.rank_map.type = UCG_RANK_MAP_TYPE_ARRAY;
    gpm.rank_map.size = static_cast<uint32_t>(nranks);
    gpm.rank_map.array = xctx->rank_map;
    gpm.oob_group.allgather = xucg_ucg_oob_cb;
    gpm.oob_group.myrank = static_cast<ucg_rank_t>(rank);
    gpm.oob_group.size = static_cast<uint32_t>(nranks);
    /* 不填 gpm.oob_group.num_local_procs：group 层该字段全树无消费方
     * （UCX 的 estimated_num_ppn 读的是 context 层那份），填了也没人看 */
    gpm.oob_group.group = xctx;

    if (ucg_group_create(g_xucg.ucg_context, &gpm, &xctx->group) != UCG_OK) {
        KuCCL_LOG_ERROR(rank, "ucg_group_create FAILED");
        return KuCCL_FAIL; /* rank_map/ctx 由 group_fini 统一回收 */
    }

    if (g_xucg.topology_ok) {
        KuCCL_LOG_INFO(rank,
                       "xucg group created (nranks=%d, group_id=%u); topo fed: nodes=%d "
                       "ppn=%u sockets=%d my_node=%d my_socket=%d | UCG pulled proc_info "
                       "x%d (with node_id x%d)",
                       nranks, xctx->group_id, g_xucg.nnode, g_xucg.num_local_procs,
                       g_xucg.socket_ok ? g_xucg.nsocket : 0,
                       g_xucg.node_id[static_cast<std::size_t>(g_xucg.global_rank)],
                       g_xucg.socket_ok ? g_xucg.socket_id[static_cast<std::size_t>(g_xucg.global_rank)] : -1,
                       g_topo_queries.load(std::memory_order_relaxed), g_topo_reported.load(std::memory_order_relaxed));
    } else {
        KuCCL_LOG_INFO(rank, "xucg group created (nranks=%d, group_id=%u); topo NOT fed (flat model)", nranks,
                       xctx->group_id);
    }
    return KuCCL_SUCCESS;
}

static void xucg_group_fini(kucclComm *comm)
{
    auto *xctx = static_cast<XucgComm *>(comm->be_ctx);
    if (xctx == nullptr)
        return;
    if (xctx->group)
        ucg_group_destroy(xctx->group);
    delete[] xctx->rank_map;
    delete xctx;
    comm->be_ctx = nullptr;
}

static void xucg_comm_purge(kucclComm *comm)
{
    purge_persistent_for_comm(comm);
}

static kucclResult_t xucg_flush(kucclComm *comm)
{
    (void)comm;
    if (g_xucg.ucp_worker) {
        ucp_worker_flush(g_xucg.ucp_worker);
    }
    return KuCCL_SUCCESS;
}

/* ════════════════════════════════════════════════════════════
 * 操作表注册
 * ════════════════════════════════════════════════════════════*/

const Ops *xucg_ops()
{
    static const Ops ops = {
        "xucg",           &xucg_process_init, &xucg_process_finalize, &xucg_group_init, &xucg_group_fini,
        &xucg_coll_start, &xucg_req_test,     &xucg_req_cancel,       &xucg_comm_purge, &xucg_flush,
    };
    return &ops;
}

} /* namespace backend */
} /* namespace kuccl */
