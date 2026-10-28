#!/usr/bin/env bash
# bench_a2a_mpi.sh — native vs direct alltoall（同参数 A/B）
#
# 两种算法跨界流量相同（都在下界上），差别在拷回/黑盒/fence 开销。
# 布局仅保留 same-die / cross-socket 两个（算法对拓扑不敏感，locality 对照用）。
# 每档按来源/目的 rank 和元素下标校验，任何算法失败即退出并保留完整日志。
#
# 用法（在 examples/ 目录）：
#   ./bench_a2a_mpi.sh [world_size]

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
LOG_DIR="./logs/a2a_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "════════════════════════════════════════════════════════"
echo " alltoall: native vs direct  ws=${WS}  per-rank sizes=2^${MIN_SHIFT}~2^${MAX_SHIFT}B"
echo " 结果: ${LOG_DIR}/"
echo "════════════════════════════════════════════════════════"

mpicxx bench_a2a_mpi.cpp -I../include -L../build -lkuccl -o bench_a2a_mpi

run() {
    local np=$1 tag=$2 algo=$3
    shift 3
    echo "  ${tag} (np=${np})"
    set +e
    KUCCL_BACKEND_TYPE=kupl KUCCL_LOG_LEVEL=INFO \
    KUCCL_KUPL_A2A_ALGO="${algo}" \
    KUCCL_LOG_FILE="${LOG_DIR}/${tag}_rank%r.log" \
    mpirun --allow-run-as-root -np "${np}" --mca coll ^ucg --report-bindings \
        -x KUCCL_BACKEND_TYPE -x KUCCL_KUPL_A2A_ALGO \
        -x KUCCL_LOG_LEVEL -x KUCCL_LOG_FILE -x UCX_WARN_UNUSED_ENV_VARS \
        "$@" ./bench_a2a_mpi "${MIN_SHIFT}" "${MAX_SHIFT}" "${ITERS}" "${WARMUP}" \
        > "${LOG_DIR}/${tag}.txt" 2>&1
    local rc=$?
    set -e
    if [ "${rc}" -eq 0 ] && grep -q "^VERIFY: ALL PASSED$" "${LOG_DIR}/${tag}.txt"; then
        echo "  ✓ ${tag}"
        return 0
    fi
    echo "  ✗ ${tag}: FAILED (rc=${rc}) — ${LOG_DIR}/${tag}.txt 与 rank 日志"
    exit 1
}

SAME=(--map-by ppr:${WS}:numa --bind-to core)
SOCK=(--map-by socket --bind-to core)

# ── 场景 1：same-die（本征开销对照，无跨界）──
run "$WS" same-native native "${SAME[@]}"
run "$WS" same-direct direct "${SAME[@]}"

# ── 场景 2：cross-socket（跨界流量对照）──
run "$WS" sock-native native "${SOCK[@]}"
run "$WS" sock-direct direct "${SOCK[@]}"

# ── 汇总 ──

SIZES=(8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)
LABELS=(8B 16B 32B 64B 128B 256B 512B 1K 2K 4K 8K 16K 32K 64K 128K 256K 512K 1M)

get() { awk -v s="$2" '$1==s+0 {print $2}' "${LOG_DIR}/$1.txt" 2>/dev/null | head -1 || true; }

pct() { # base val → Δ%（正 = 更快）
    awk -v b="$1" -v v="$2" 'BEGIN{ if (b+0 > 0) printf "%+d%%", (b-v)/b*100; else printf "—" }'
}

cell() { [ -n "$1" ] && printf " %10.1f" "$1" || printf " %10s" "—"; }

echo ""
echo "──────────────────────────────────────────────────────────────────────────"
echo " same-die — 本征开销对照（无跨界）"
echo "──────────────────────────────────────────────────────────────────────────"
printf "%-8s %10s %10s %8s\n" "size" "native" "direct" "Δdirect"
for i in "${!SIZES[@]}"; do
    s=${SIZES[$i]}
    n=$(get same-native "$s"); d=$(get same-direct "$s")
    d1="—"; [ -n "$n" ] && [ -n "$d" ] && d1=$(pct "$n" "$d")
    printf "%-8s" "${LABELS[$i]}"; cell "$n"; cell "$d"; printf " %8s\n" "$d1"
done

echo ""
echo "──────────────────────────────────────────────────────────────────────────"
echo " cross-socket — 跨界流量对照"
echo "──────────────────────────────────────────────────────────────────────────"
printf "%-8s %10s %10s %8s\n" "size" "native" "direct" "Δdirect"
for i in "${!SIZES[@]}"; do
    s=${SIZES[$i]}
    n=$(get sock-native "$s"); d=$(get sock-direct "$s")
    d1="—"; [ -n "$n" ] && [ -n "$d" ] && d1=$(pct "$n" "$d")
    printf "%-8s" "${LABELS[$i]}"; cell "$n"; cell "$d"; printf " %8s\n" "$d1"
done

echo ""
echo "════════════════════════════════════════════════════════"
echo " 说明："
echo "   native = KUPL 原生 alltoall（staging → 黑盒 → 拷回全量）"
echo "   direct = 直拉（一次 staging + 2 次 fence，直写用户 recvbuf）"
echo "   两种算法跨界流量相同（都在下界上）——Δ 来自拷回/黑盒/fence 开销"
echo "   Δ 正值 = 比 native 快；每档已逐元素 VERIFY"
echo "   ★ native/direct 均要求全部 rank 校验通过且 mpirun 成功退出"
echo " 结果: ${LOG_DIR}/"
echo "════════════════════════════════════════════════════════"

cat > "${LOG_DIR}/meta.txt" << EOF
timestamp: ${TIMESTAMP}
world_size: ${WS}
git: $(git -C ../.. rev-parse --short HEAD 2>/dev/null || echo unknown)
layouts:
  same-die:     ${SAME[*]}
  cross-socket: ${SOCK[*]}
sizes: 2^${MIN_SHIFT} ~ 2^${MAX_SHIFT} bytes per rank
iters: ${ITERS}  warmup: ${WARMUP}
op: alltoall int32 (send[src][dst][m] = (src*W+dst)*count+m)
EOF
