#!/usr/bin/env bash
# ppn16_report.sh —— 单节点 PPN=16 kupl vs gloo 性能报告
#
# 一条命令产出：逐尺寸延迟对比表 + 提升百分比 + 判定（≥30%），
# 结果带时间戳/git HEAD 存档。
#
# 用法（先 source 环境并完成两包构建，从仓库根出发）：
#   cd csrc/kuccl && source ./kuccl_env.sh && ./build.sh
#   cd ../torch_kuccl && source env.sh && ./build.sh && pip install -e . --no-build-isolation
#   cd ../../benchmarks/communication && ./ppn16_report.sh
#
# 可调（环境变量）：
#   WS       world_size（默认 16，即 PPN=16）
#   OPS      算子列表（默认四个集合算子全跑）
#   TRIALS   每档测量轮数（默认 30）
#   MAXSIZE  最大消息档位（2^MAXSIZE 元素，默认 19）
#   GATE     判定阈值 %（默认 30）
#   OUT      报告输出文件名（默认带时间戳）

set -uo pipefail
cd "$(dirname "$0")"

WS=${WS:-16}
OPS=${OPS:-"all_reduce all_gather broadcast all_to_all"}
TRIALS=${TRIALS:-30}
MAXSIZE=${MAXSIZE:-19}
GATE=${GATE:-30}
OUT=${OUT:-"perf_report_ppn${WS}_$(date +%Y%m%d_%H%M%S).md"}

export KUCCL_BACKEND_TYPE=${KUCCL_BACKEND_TYPE:-kupl}
HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

{
    echo "# kuccl 单节点 PPN=${WS} 性能报告（kupl vs gloo，目标 ≥${GATE}%）"
    echo
    echo "- 时间: $(date '+%F %T')"
    echo "- git HEAD: ${HEAD}"
    echo "- KUCCL_BACKEND_TYPE: ${KUCCL_BACKEND_TYPE}"
    echo "- 参数: ops=[${OPS}] trials=${TRIALS} maxsize=2^${MAXSIZE} gate=${GATE}%"
    echo
    python compare_run.py --ops $OPS --world-size "$WS" \
        --trials "$TRIALS" --maxsize "$MAXSIZE" --gate "$GATE"
} 2>&1 | tee "$OUT"

echo
echo "[报告已保存] $(pwd)/$OUT"
echo "── 判定摘要 ──"
grep -E "提升|gate|→" "$OUT" | tail -10 || true
