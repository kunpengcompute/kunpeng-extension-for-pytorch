#!/usr/bin/env bash
# bench_ar_compare.sh — allreduce 全面对比
#
# 统一口径（全部同参数，差异只来自实现）：
#   ws, sizes 2^3~2^20（18 档）, iters=trials, warmup=5, 三布局（same / die / socket）
# 矩阵（每布局 6 列）：
#   mpi-ar-flat = kupl 原生 allreduce（KUPL 黑盒）
#   mpi-ar-hier   = 分层实现（组内 RS → o/e 子块交换 → 收账）
#   mpi-ar-auto   = 生产路由（flat 小 / hier ≥ KUCCL_KUPL_AR_HIER_MIN）
#   mpi-xucg      = xucg 后端参照（可选列，失败只警告）
#   torch-kupl    = torch 适配层 kupl（auto 路由）
#   torch-gloo    = torch gloo 基线（不绑核，对照用；三布局共用一份）
# 汇总：每布局一张大表（us）+ Δhier%（算法口径）+ Δprod%（vs gloo，目标
#   ≥30%）+ 绑核一致性核对
# VERIFY：kupl 三列 FAIL → 非零退出；xucg 只警告。
# 结果：./logs/ar_compare/<时间戳>/compare-<layout>.txt + meta.txt
#
# 用法（在 examples/ 目录）：
#   ./bench_ar_compare.sh [world_size]
#   LAYOUTS="die socket" ./bench_ar_compare.sh 16   # 只跑部分布局
#   REPEATS=3 STAT=min ./bench_ar_compare.sh 16     # min 抗抖动

set -euo pipefail
cd "$(dirname "$0")"
source ../kuccl_env.sh

WS=${1:-16}
MIN_SHIFT=3
MAX_SHIFT=20
TRIALS=${TRIALS:-30}
ITERS=$TRIALS          # MPI 侧与 torch 侧同重复次数——口径强制一致
WARMUP=5
REPEATS=${REPEATS:-1}  # 每 run 重复次数，汇总按 STAT 聚合
STAT=${STAT:-avg}      # avg（算术平均）/ min（抗抖动，推荐 min + REPEATS=3）
TORCH_COMM=../../../benchmarks/communication
PORT=29740
export UCX_WARN_UNUSED_ENV_VARS=n
LAYOUTS=${LAYOUTS:-"same die socket"}

# ── 预检 1：die 布局要求偶数 WS ──
if [ $((WS % 2)) -ne 0 ]; then
    echo "✗ die 布局需要偶数 world_size（当前 WS=${WS}）"; exit 1
fi

# ── 预检 2：残留进程/高负载会污染数据 ──
LEFT=$(ps -eo args 2>/dev/null | grep -E "mpirun|correctness\.py|bench_(ag|ar)_mpi|run_cpu\.py" | grep -v grep || true)
if [ -n "${LEFT}" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "✗ 检测到疑似残留进程（会抢核污染数据）："
    echo "${LEFT}"
    echo "  清理：pkill -9 -f correctness.py; pkill -9 -f mpirun   或 FORCE=1 强行继续"
    exit 1
fi
LOAD1=$(cut -d' ' -f1 /proc/loadavg)
NPROC=$(nproc)
if awk -v l="${LOAD1}" -v n="${NPROC}" 'BEGIN{exit !(l > n * 0.8)}' && [ "${FORCE:-0}" != "1" ]; then
    echo "✗ 负载偏高（load1=${LOAD1} / ${NPROC} 核）——先清场再测（或 FORCE=1）"
    exit 1
fi

# ── 预检 2b：torch 列需要 conda 环境（无 torch 时只警告，MPI 列照跑）──
if ! python -c "import torch" >/dev/null 2>&1; then
    echo "⚠ 当前 python 无 torch——torch-kupl/torch-gloo 列将缺数据（先 conda activate kuccl 再跑可补全）"
fi

TS=$(date +%Y%m%d_%H%M%S)
OUT=./logs/ar_compare/${TS}_t${TRIALS}
mkdir -p "$OUT"

echo "════════════════════════════════════════════════════════"
echo " allreduce 全面对比 v2  ws=${WS}  sizes=2^${MIN_SHIFT}~2^${MAX_SHIFT}B  iters=${ITERS}  repeats=${REPEATS}(${STAT})"
echo " 矩阵: mpi{ar-flat, ar-hier, ar-auto, xucg} + torch{kupl-auto, gloo} × {${LAYOUTS}}"
echo " 结果: ${OUT}/"
echo "════════════════════════════════════════════════════════"

mpicxx bench_ar_mpi.cpp -I../include -L../build -lkuccl -o bench_ar_mpi

# ── MPI 侧：直调 kuccl C API ──
run_mpi() { # tag kuccl-backend ar-algo-or-'-' map...
    local tag=$1 be=$2 algo=$3
    shift 3
    local k rc
    for k in $(seq 1 "${REPEATS}"); do
        local f="${OUT}/${tag}-r${k}.txt"
        local extra=()
        [ "${algo}" != "-" ] && extra+=(-x KUCCL_KUPL_AR_ALGO)
        [ -n "${KUCCL_KUPL_AR_HIER_MIN:-}" ] && extra+=(-x KUCCL_KUPL_AR_HIER_MIN)
        set +e
        env KUCCL_BACKEND_TYPE="$be" KUCCL_LOG_LEVEL=INFO \
        ${algo:+KUCCL_KUPL_AR_ALGO=${algo}} \
        KUCCL_LOG_FILE="${OUT}/${tag}-r${k}_rank%r.log" \
        mpirun --allow-run-as-root -np "${WS}" --mca coll ^ucg --report-bindings \
            -x KUCCL_BACKEND_TYPE -x KUCCL_LOG_LEVEL -x KUCCL_LOG_FILE \
            -x UCX_WARN_UNUSED_ENV_VARS ${extra[@]+"${extra[@]}"} \
            "$@" ./bench_ar_mpi "${MIN_SHIFT}" "${MAX_SHIFT}" "${ITERS}" "${WARMUP}" \
            2>&1 | grep -v "kupl_sdma\|kupl_memcpy\|SDMA ERROR\|UCX  WARN" \
            > "$f"
        rc=$?
        set -e
        if [ "$rc" != "0" ] || ! grep -q "VERIFY: ALL PASSED" "$f"; then
            echo "  [warn] ${tag}-r${k} 失败（exit=${rc}，该列将缺数据）— $f tail:"
            tail -6 "$f"
            return 1
        fi
    done
    return 0
}

# ── torch 侧：run_cpu.py 拉起（生产口径：OMP=1）──
run_torch() { # tag torch-backend kuccl-backend-or-'-' bind-env-or-'-'
    local tag=$1 tbe=$2 kbe=$3 bind=$4
    if [ "$kbe" = "-" ]; then unset KUCCL_BACKEND_TYPE; else export KUCCL_BACKEND_TYPE="$kbe"; fi
    unset KUCCL_KUPL_AR_ALGO || true        # torch-kupl 恒走 auto（生产路由）
    if [ "$bind" != "-" ]; then export KUCCL_BIND_LAYOUT="$bind"; else unset KUCCL_BIND_LAYOUT; fi
    export OMP_NUM_THREADS=1
    local k rc
    for k in $(seq 1 "${REPEATS}"); do
        PORT=$((PORT + 1))
        local f="${OUT}/${tag}-r${k}.txt"
        set +e
        python "${TORCH_COMM}/run_cpu.py" --world-size "${WS}" \
            --master-addr 127.0.0.1 --master-port "${PORT}" \
            -- all_reduce.py --dist torch --backend "${tbe}" --device cpu \
            --scan --maxsize 20 --trials "${TRIALS}" --warmups "${WARMUP}" --raw \
            > "$f" 2>&1
        rc=$?
        set -e
        if [ "$rc" != "0" ]; then
            echo "  [warn] ${tag}-r${k} 失败（exit=${rc}，该列将缺数据）— $f tail:"
            tail -6 "$f"
            return 1
        fi
    done
    return 0
}

LAYOUT_MPI_same=(--map-by ppr:${WS}:numa --bind-to core)
LAYOUT_MPI_die=(--map-by ppr:$((WS / 2)):numa --bind-to core)
LAYOUT_MPI_socket=(--map-by socket --bind-to core)
LAYOUT_TORCH_same="die:${WS}"
LAYOUT_TORCH_die="die:$((WS / 2))"
LAYOUT_TORCH_socket="socket"

echo "  torch-gloo"
run_torch "torch-gloo" gloo "-" "-" || true

for L in ${LAYOUTS}; do
    case $L in
        same)   MPI_MAP=("${LAYOUT_MPI_same[@]}");   TBIND=$LAYOUT_TORCH_same;;
        die)    MPI_MAP=("${LAYOUT_MPI_die[@]}");    TBIND=$LAYOUT_TORCH_die;;
        socket) MPI_MAP=("${LAYOUT_MPI_socket[@]}"); TBIND=$LAYOUT_TORCH_socket;;
        *) echo "unknown layout: $L"; exit 1;;
    esac
    echo "── ${L} ──"
    echo "  mpi-ar-flat"; run_mpi "mpi-${L}-arflat" kupl flat "${MPI_MAP[@]}" || true
    echo "  mpi-ar-hier";  run_mpi "mpi-${L}-arhier"  kupl hier  "${MPI_MAP[@]}" || true
    echo "  mpi-ar-auto";  run_mpi "mpi-${L}-arauto"  kupl auto  "${MPI_MAP[@]}" || true
    echo "  mpi-xucg";      run_mpi "mpi-${L}-xucg"    xucg -     "${MPI_MAP[@]}" || true
    echo "  torch-kupl";    run_torch "torch-${L}-kupl" kuccl kupl "$TBIND" || true

    python - "$OUT" "$TORCH_COMM" "$L" "$STAT" <<'EOF'
import glob
import os
import re
import sys

out_dir, torch_comm, layout, stat = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, os.path.abspath(torch_comm))
from compare_run import parse_raw_output


def parse_mpi(path):
    d = {}
    if not os.path.exists(path):
        return d
    for line in open(path):
        m = re.match(r'\s*(\d+)\s+([\d.]+)\s', line)
        if m:
            d[int(m.group(1))] = float(m.group(2))
    return d


def parse_torch(path):
    if not os.path.exists(path):
        return {}
    return parse_raw_output(open(path).read(), 'all_reduce')


def best(parse, prefix):
    """prefix-r*.txt 全部重复轮次按 STAT 聚合：min（抗抖动）/ avg（算术平均）"""
    files = sorted(glob.glob(os.path.join(out_dir, prefix + '-r*.txt')))
    if not files:
        return {}
    if stat == 'avg':
        sums, cnts = {}, {}
        for p in files:
            for s, v in parse(p).items():
                sums[s] = sums.get(s, 0.0) + v
                cnts[s] = cnts.get(s, 0) + 1
        return {s: sums[s] / cnts[s] for s in sums}
    d = {}
    for p in files:
        for s, v in parse(p).items():
            d[s] = min(d.get(s, float('inf')), v)
    return d


def parse_mpi_cores(path):
    d = {}
    if not os.path.exists(path):
        return d
    rank = None
    for line in open(path):
        m = re.search(r'(?:MCW )?rank (\d+) bound to', line)
        if m:
            rank = int(m.group(1))
            m2 = re.search(r'\[core (\d+)\[', line)
            if m2 and rank not in d:
                d[rank] = int(m2.group(1))
            continue
        m = re.search(r'Data for rank (\d+):', line)
        if m:
            rank = int(m.group(1))
            continue
        if rank is not None and rank not in d:
            m = re.search(r'\bC([0-9a-fA-F]{2,6})\b', line)
            if m:
                d[rank] = int(m.group(1), 16)
    return d


def parse_torch_cores(path):
    d = {}
    if not os.path.exists(path):
        return d
    for line in open(path):
        m = re.match(r'\s*\[bind\] rank (\d+) -> node \d+ \(usable cores \d+, e\.g\. (\d+)\)', line)
        if m:
            d[int(m.group(1))] = int(m.group(2))
    return d


def binding_status():
    mc = parse_mpi_cores(os.path.join(out_dir, f'mpi-{layout}-arflat-r1.txt'))
    tc = parse_torch_cores(os.path.join(out_dir, f'torch-{layout}-kupl-r1.txt'))
    if not mc or not tc:
        return 'bindings: UNVERIFIED (missing report)'
    if mc == tc:
        return f'bindings: MATCH — {len(mc)} ranks, e.g. rank0=core{mc.get(0, "?")}'
    diff = [r for r in sorted(set(mc) | set(tc)) if mc.get(r) != tc.get(r)]
    sample = ', '.join(f'r{r}: mpi={mc.get(r)} torch={tc.get(r)}' for r in diff[:4])
    return f'bindings: MISMATCH — {len(diff)} ranks differ ({sample}...)'


COLS = [
    ('mpi-flat',     best(parse_mpi, f'mpi-{layout}-arflat')),
    ('mpi-hier',     best(parse_mpi, f'mpi-{layout}-arhier')),
    ('mpi-auto',     best(parse_mpi, f'mpi-{layout}-arauto')),
    ('mpi-xucg',     best(parse_mpi, f'mpi-{layout}-xucg')),
    ('torch-kupl',   best(parse_torch, f'torch-{layout}-kupl')),
    ('torch-gloo',   best(parse_torch, 'torch-gloo')),
]

sizes = sorted({s for _, d in COLS for s in d})


def fmt(v):
    return f'{v:10.1f}' if v else f'{"—":>10}'


def pct(new, base):
    """正 = new 比 base 快（us 口径）"""
    if new and base and base > 0:
        return f'{(base - new) / base * 100:>7.0f}%'
    return f'{"—":>8}'


lines = [f'layout: {layout}  stat: {stat}',
         binding_status(),
         f'{"size":>10}' + ''.join(f'{n:>12}' for n, _ in COLS) + f'{"Δhier%":>9}{"Δprod%":>9}',
         '-' * (10 + 12 * len(COLS) + 18)]
for s in sizes:
    row = f'{s:>10}'
    for _, d in COLS:
        row += fmt(d.get(s))
    row += pct(COLS[1][1].get(s), COLS[0][1].get(s))     # Δhier%  = hier vs flat
    row += pct(COLS[4][1].get(s), COLS[5][1].get(s))     # Δprod%  = torch-kupl vs torch-gloo（目标 ≥30%）
    lines.append(row)
table = '\n'.join(lines)
print(table)
with open(os.path.join(out_dir, f'compare-{layout}.txt'), 'w') as f:
    f.write(table + '\n')
EOF

done

# ── 正确性检查（收尾统一裁决）──
# kupl 三列（flat/hier/auto）= 我们自己的路径，VERIFY FAIL 即非零退出；
# xucg = 第三方参照，失败只警告。
BAD=0
XBAD=0
for col in arflat arhier arauto; do
    for f in "${OUT}"/mpi-*-"${col}"-r*.txt; do
        [ -e "$f" ] || continue
        if ! grep -q "VERIFY: ALL PASSED" "$f"; then
            echo "✗ VERIFY FAIL（kupl 列）: $f"
            BAD=1
        fi
    done
done
for f in "${OUT}"/mpi-*-xucg-r*.txt; do
    [ -e "$f" ] || continue
    if ! grep -q "VERIFY: ALL PASSED" "$f"; then
        echo "⚠ VERIFY FAIL（xucg 参照列，仅警告）: $f"
        XBAD=1
    fi
done

# ── meta 记录（复现凭据）──
cat > "${OUT}/meta.txt" << EOF
timestamp: ${TS}
git: $(git -C ../.. rev-parse --short HEAD 2>/dev/null || echo unknown)
world_size: ${WS}  sizes: 2^${MIN_SHIFT}~2^${MAX_SHIFT}B  iters: ${ITERS}  trials: ${TRIALS}
warmup: ${WARMUP}  repeats: ${REPEATS}  stat: ${STAT}
layouts: ${LAYOUTS}
matrix: mpi{ar-flat, ar-hier, ar-auto, xucg} + torch{kupl-auto, gloo}
KUCCL_KUPL_AR_HIER_MIN: ${KUCCL_KUPL_AR_HIER_MIN:-unset}
Δhier%  = (flat−hier)/flat （算法口径，正 = 分层更快）
Δprod%  = (gloo−torch-kupl)/gloo（目标 ≥30%）
EOF

echo ""
echo "════════════════════════════════════════════════════════"
echo " 说明："
echo "   mpi-flat = kupl 原生（KUPL 黑盒）；mpi-hier = 分层；mpi-auto = 生产路由"
echo "   mpi-xucg = xucg 参照；torch-kupl = auto 路由；torch-gloo = 对照基线"
echo "   单位 us；Δhier% 正 = 分层比原生快；Δprod% 正 = kupl 比 gloo 快（目标 ≥30%）"
if [ "${XBAD}" != "0" ]; then echo "   ⚠ xucg 参照列有 VERIFY FAIL（见上方 ⚠ 行）"; fi
echo "   VERIFY：kupl 列 FAIL = ${BAD}（0 = 全绿）；meta/汇总见 ${OUT}/"
echo "════════════════════════════════════════════════════════"
exit "${BAD}"
