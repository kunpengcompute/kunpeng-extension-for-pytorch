#!/usr/bin/env bash
# bench_ar_mpi.sh — flat vs 分层 allreduce（部署场景 × 算法）
#
# 回答一个问题：分层 allreduce 在什么部署环境下赢 flat？auto（尺寸路由）与显式列是否一致？
#   same-die     全部 rank 在一个 NUMA 节点     — 无分层边界（安全网：应与 flat 持平）
#   cross-die    前 2 个 NUMA 节点各半         — hier_die 的用武之地
#   cross-socket 2 个 socket 各半              — hier_socket 的用武之地
#
# 每个场景内 mpirun 布局固定，唯一变量是算法（KUCCL_KUPL_AR_ALGO）。
# 每档逐元素 VERIFY（Σrank mod 256），挂了判失败。
# 结果存放于 ./logs/<时间戳>/（与脚本同级，不进仓库目录）。
#
# 用法（在 examples/ 目录）：
#   ./bench_ar_mpi.sh [world_size]

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
LOG_DIR="./logs/ar_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "════════════════════════════════════════════════════════"
echo " allreduce: flat vs 分层  ws=${WS}  sizes=2^${MIN_SHIFT}~2^${MAX_SHIFT}B"
echo " 结果: ${LOG_DIR}/"
echo "════════════════════════════════════════════════════════"

mpicxx bench_ar_mpi.cpp -I../include -L../build -lkuccl -o bench_ar_mpi

run() {
    local np=$1 tag=$2 algo=$3
    shift 3
    echo "  ${tag} (np=${np})"
    set +e
    KUCCL_BACKEND_TYPE=kupl KUCCL_LOG_LEVEL=INFO \
    KUCCL_KUPL_AR_ALGO="${algo}" \
    KUCCL_LOG_FILE="${LOG_DIR}/${tag}_rank%r.log" \
    mpirun --allow-run-as-root -np "${np}" --mca coll ^ucg --report-bindings \
        -x KUCCL_BACKEND_TYPE -x KUCCL_KUPL_AR_ALGO \
        -x KUCCL_LOG_LEVEL -x KUCCL_LOG_FILE -x UCX_WARN_UNUSED_ENV_VARS \
        "$@" ./bench_ar_mpi "${MIN_SHIFT}" "${MAX_SHIFT}" "${ITERS}" "${WARMUP}" \
        2>&1 | grep -v "kupl_sdma\|kupl_memcpy\|SDMA ERROR\|UCX  WARN" \
        > "${LOG_DIR}/${tag}.txt"
    local rc=$?
    set -e
    if [ "$rc" != "0" ] || ! grep -q "VERIFY: ALL PASSED" "${LOG_DIR}/${tag}.txt"; then
        echo "  ✗ ${tag}: FAILED（往上翻 ${LOG_DIR}/${tag}.txt 与 rank 日志）"
        exit 1
    fi
}

# 场景布局（mpirun 拓扑感知映射）：
#   same-die     ppr:WS:numa      → 全部塞进第 1 个 NUMA 节点
#   cross-die    ppr:(WS/2):numa  → 前 2 个 NUMA 节点各 WS/2
#   cross-socket socket           → 2 个 socket 各 WS/2
SAME=(--map-by ppr:${WS}:numa --bind-to core)
DIE2=(--map-by ppr:$((WS / 2)):numa --bind-to core)
SOCK=(--map-by socket --bind-to core)

# ── 场景 1：same-die（无分层边界 — hier 应退回 flat，安全网验证）──
run "$WS" same-flat flat "${SAME[@]}"
run "$WS" same-hier  hier  "${SAME[@]}"
run "$WS" same-auto  auto  "${SAME[@]}"

# ── 场景 2：cross-die（hier 自动选 die 粒度）──
run "$WS" die-flat flat "${DIE2[@]}"
run "$WS" die-hier  hier  "${DIE2[@]}"
run "$WS" die-auto  auto  "${DIE2[@]}"

# ── 场景 3：cross-socket（hier 自动选 socket 粒度）──
run "$WS" sock-flat flat "${SOCK[@]}"
run "$WS" sock-hier  hier  "${SOCK[@]}"
run "$WS" sock-auto  auto  "${SOCK[@]}"

# ── 场景 4：uneven（不均匀 NUMA 落位——非属主路径的确定性回归网）──
# 用 rankfile 把 rank 强制按 (WS/2+1)+(WS/2−1) 摆到前两个 NUMA 节点：
# 大组中 idx ≥ M=min(组大小) 的成员不持切片、只收 ③ 扇出。回归 85614e5——
# 非属主曾拿到越界"幽灵切片"（≤512KB 静默写段空余、1MB 冲出映射 SEGV），
# 此前 correctness 的拓扑全凭调度器随缘，均匀落位时该路径零覆盖。
UNEVEN_RF="${LOG_DIR}/rankfile_uneven.txt"
UNEVEN_A=$((WS / 2 + 1))
UNEVEN_B=$((WS - UNEVEN_A))
{
    i=0
    for c in $(lscpu -p=CPU,NODE | awk -F, '!/^#/ && $2==0 {print $1}' | head -"$UNEVEN_A"); do
        echo "rank $((i++))=localhost slot=$c"
    done
    for c in $(lscpu -p=CPU,NODE | awk -F, '!/^#/ && $2==1 {print $1}' | head -"$UNEVEN_B"); do
        echo "rank $((i++))=localhost slot=$c"
    done
} > "$UNEVEN_RF"
UNEVEN=(--rankfile "$UNEVEN_RF" --bind-to core)
run "$WS" uneven-flat flat "${UNEVEN[@]}"
run "$WS" uneven-hier  hier  "${UNEVEN[@]}"

# ── 汇总 ──

SIZES=(8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)
LABELS=(8B 16B 32B 64B 128B 256B 512B 1K 2K 4K 8K 16K 32K 64K 128K 256K 512K 1M)

get() { awk -v s="$2" '$1==s+0 {print $2}' "${LOG_DIR}/$1.txt" 2>/dev/null | head -1 || true; }

pct() { # base val → Δ%（正 = 更快）
    awk -v b="$1" -v v="$2" 'BEGIN{ if (b+0 > 0) printf "%+d%%", (b-v)/b*100; else printf "—" }'
}

cell() { [ -n "$1" ] && printf " %10.1f" "$1" || printf " %10s" "—"; }

# block <标题> <副标题> <flat_tag> <列1_tag> <列1名> <列2_tag> <列2名> <auto_tag>
# 列顺序: flat | 列1 | 列2 | auto | Δ列1 | Δ列2 | Δauto（空列自动跳过）
block() {
    local title=$1 sub=$2 flat=$3 t1=$4 n1=$5 t2=$6 n2=$7 auto=$8
    echo ""
    echo "──────────────────────────────────────────────────────────────────────────"
    echo " $title — $sub"
    echo "──────────────────────────────────────────────────────────────────────────"
    printf "%-8s %10s" "size" "flat"
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
        p=$(get "$flat" "$s")
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

block "same-die" "全部 rank 在一个 NUMA 节点（hier 应退回 flat = 安全网）" same-flat "" "" "" "" same-auto
block "cross-die" "前 2 个 NUMA 节点各半（hier 自动选 die 粒度）" die-flat die-hier "hier" "" "" die-auto
block "cross-socket" "2 个 socket 各半（hier 自动选 socket 粒度）" sock-flat sock-hier "hier" "" "" sock-auto
block "uneven" "不均匀 NUMA（${UNEVEN_A}+${UNEVEN_B}，非属主只收扇出——85614e5 回归网）" uneven-flat uneven-hier "hier" "" "" ""

echo ""
echo "════════════════════════════════════════════════════════"
echo " 说明："
echo "   auto = 尺寸路由（小消息 flat：SUM→KUPL 原生 RH-RD）——小尺寸应与 flat 列一致"
echo "   hier = 分层归约（组内 reduce-scatter → 组间交换 → 组内收账），粒度按拓扑自动选："
echo "         same-die 应回退 flat（安全网）、cross-die 选 die 粒度、cross-socket 选 socket 粒度"
echo "         （实际选择见 rank 日志 [AR-route] 行）"
echo "   预期：大消息 + cross-socket 布局收益最大（跨 socket 链路只跑已归约数据）；"
echo "         小消息 4 次 fence 会亏——这正是 opt-in 的原因"
echo "   Δ 正值 = 比 flat 快；每档已逐元素 VERIFY"
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
op: allreduce SUM uint8 (verify: sum(rank) mod 256)
EOF
