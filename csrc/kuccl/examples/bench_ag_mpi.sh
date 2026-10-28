#!/usr/bin/env bash
# bench_ag_mpi.sh — linear vs 分层 allgather（部署场景 × 算法）
#
# 回答一个问题：分层算法在什么部署环境下赢 linear？auto（拓扑合适→分层，
# 不合适→linear）与显式分层列是否一致？
#   same-die     全部 rank 在一个 NUMA 节点     — 无分层边界，auto 应与 linear 持平
#   cross-die    前 2 个 NUMA 节点各半         — hier_die 的用武之地，auto 应选中它
#   cross-socket 2 个 socket 各半              — hier_socket 的用武之地，auto 应选中它
#
# 每个场景内 mpirun 布局固定，唯一变量是算法。
# 结果存放于 ./logs/<时间戳>/（与脚本同级，不进仓库目录）。
#
# 用法（在 examples/ 目录）：
#   ./bench_ag_mpi.sh [world_size]

set -euo pipefail
cd "$(dirname "$0")"
source ../kuccl_env.sh

WS=${1:-16}
MIN_SHIFT=3
MAX_SHIFT=20
ITERS=30
WARMUP=5
export UCX_WARN_UNUSED_ENV_VARS=n

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "════════════════════════════════════════════════════════"
echo " allgather: linear vs 分层  ws=${WS}  sizes=2^${MIN_SHIFT}~2^${MAX_SHIFT}B"
echo " 结果: ${LOG_DIR}/"
echo "════════════════════════════════════════════════════════"

mpicxx bench_ag_mpi.cpp -I../include -L../build -lkuccl -o bench_ag_mpi

run() {
    local np=$1 tag=$2 algo=$3
    shift 3
    echo "  ${tag} (np=${np})"
    KUCCL_BACKEND_TYPE=kupl KUCCL_LOG_LEVEL=INFO \
    KUCCL_KUPL_AG_ALGO="${algo}" \
    KUCCL_LOG_FILE="${LOG_DIR}/${tag}_rank%r.log" \
    mpirun --allow-run-as-root -np "${np}" --mca coll ^ucg --report-bindings \
        -x KUCCL_BACKEND_TYPE -x KUCCL_KUPL_AG_ALGO \
        -x KUCCL_LOG_LEVEL -x KUCCL_LOG_FILE -x UCX_WARN_UNUSED_ENV_VARS \
        "$@" ./bench_ag_mpi "${MIN_SHIFT}" "${MAX_SHIFT}" "${ITERS}" "${WARMUP}" \
        2>&1 | grep -v "kupl_sdma\|kupl_memcpy\|SDMA ERROR\|UCX  WARN" \
        > "${LOG_DIR}/${tag}.txt"
}

# 场景布局（mpirun 拓扑感知映射）：
#   same-die     ppr:WS:numa      → 全部塞进第 1 个 NUMA 节点
#   cross-die    ppr:(WS/2):numa  → 前 2 个 NUMA 节点各 WS/2
#   cross-socket socket           → 2 个 socket 各 WS/2
SAME=(--map-by ppr:${WS}:numa --bind-to core)
DIE2=(--map-by ppr:$((WS / 2)):numa --bind-to core)
SOCK=(--map-by socket --bind-to core)

# ── 场景 1：same-die（无分层边界 — 安全网验证）──
run "$WS" same-linear linear "${SAME[@]}"
run "$WS" same-auto   auto   "${SAME[@]}"

# ── 场景 2：cross-die（hier_die 的用武之地）──
run "$WS" die-linear  linear  "${DIE2[@]}"
run "$WS" die-hierdie hier_die "${DIE2[@]}"
run "$WS" die-auto    auto     "${DIE2[@]}"

# ── 场景 3：cross-socket（die 分组与 socket 分组重合，hier_socket 主战）──
run "$WS" sock-linear  linear       "${SOCK[@]}"
run "$WS" sock-hierdie hier_die     "${SOCK[@]}"
run "$WS" sock-hierskt hier_socket  "${SOCK[@]}"
run "$WS" sock-auto    auto         "${SOCK[@]}"

# ── 汇总 ──

SIZES=(8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)
LABELS=(8B 16B 32B 64B 128B 256B 512B 1K 2K 4K 8K 16K 32K 64K 128K 256K 512K 1M)

get() { awk -v s="$2" '$1==s+0 {print $2}' "${LOG_DIR}/$1.txt" 2>/dev/null | head -1 || true; }

pct() { # base val → Δ%（正 = 更快）
    awk -v b="$1" -v v="$2" 'BEGIN{ if (b+0 > 0) printf "%+d%%", (b-v)/b*100; else printf "—" }'
}

cell() { [ -n "$1" ] && printf " %10.1f" "$1" || printf " %10s" "—"; }

# block <标题> <副标题> <linear_tag> <列1_tag> <列1名> <列2_tag> <列2名> <auto_tag>
# 列顺序: linear | 列1 | 列2 | auto | Δ列1 | Δ列2 | Δauto（空列自动跳过）
block() {
    local title=$1 sub=$2 plain=$3 t1=$4 n1=$5 t2=$6 n2=$7 auto=$8
    echo ""
    echo "──────────────────────────────────────────────────────────────────────────"
    echo " $title — $sub"
    echo "──────────────────────────────────────────────────────────────────────────"
    printf "%-8s %10s" "size" "linear"
    if [ -n "$n1" ]; then printf " %10s" "$n1"; fi
    if [ -n "$n2" ]; then printf " %10s" "$n2"; fi
    printf " %10s" "auto"
    if [ -n "$n1" ]; then printf " %8s" "Δ$n1"; fi
    if [ -n "$n2" ]; then printf " %8s" "Δ$n2"; fi
    printf " %8s\n" "Δauto"
    local i s L p a v1 v2 dA d1 d2
    for i in "${!SIZES[@]}"; do
        s=${SIZES[$i]}
        L=${LABELS[$i]}
        p=$(get "$plain" "$s")
        a=$(get "$auto" "$s")
        v1=""
        v2=""
        if [ -n "$t1" ]; then v1=$(get "$t1" "$s"); fi
        if [ -n "$t2" ]; then v2=$(get "$t2" "$s"); fi
        dA="—"
        d1="—"
        d2="—"
        if [ -n "$p" ] && [ -n "$a" ]; then dA=$(pct "$p" "$a"); fi
        if [ -n "$p" ] && [ -n "$v1" ]; then d1=$(pct "$p" "$v1"); fi
        if [ -n "$p" ] && [ -n "$v2" ]; then d2=$(pct "$p" "$v2"); fi
        printf "%-8s" "$L"
        cell "$p"
        if [ -n "$n1" ]; then cell "$v1"; fi
        if [ -n "$n2" ]; then cell "$v2"; fi
        cell "$a"
        if [ -n "$n1" ]; then printf " %8s" "$d1"; fi
        if [ -n "$n2" ]; then printf " %8s" "$d2"; fi
        printf " %8s\n" "$dA"
    done
}

block "same-die" "全部 rank 在一个 NUMA 节点（无分层边界）" same-linear "" "" "" "" same-auto
block "cross-die" "前 2 个 NUMA 节点各半（同 socket 跨 die）" die-linear die-hierdie "hier_die" "" "" die-auto
block "cross-socket" "2 个 socket 各半（die 分组与 socket 分组重合）" sock-linear sock-hierdie "hier_die" sock-hierskt "hier_socket" sock-auto

echo ""
echo "════════════════════════════════════════════════════════"
echo " 说明："
echo "   auto = 拓扑路由（有分层边界→分层，否则 linear）——same-die 应与 linear 持平，"
echo "         cross-die/cross-socket 应与对应分层列一致"
echo "   hier_die 按 die 配对；hier_socket 跨 socket 配对（1 个远读搭档 + 本组 fanout）"
echo "   Δ 正值 = 比 linear 快"
echo " 结果: ${LOG_DIR}/"
echo "════════════════════════════════════════════════════════"

# 记录元数据
cat > "${LOG_DIR}/meta.txt" << EOF
timestamp: ${TIMESTAMP}
world_size: ${WS}
git: $(git -C ../.. rev-parse --short HEAD 2>/dev/null || echo unknown)
layouts:
  same-die:     ${SAME[*]}
  cross-die:    ${DIE2[*]}
  cross-socket: ${SOCK[*]}
sizes: 2^${MIN_SHIFT} ~ 2^${MAX_SHIFT} bytes
iters: ${ITERS}  warmup: ${WARMUP}
EOF
