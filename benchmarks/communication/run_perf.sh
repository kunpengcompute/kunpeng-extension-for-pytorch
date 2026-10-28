#!/usr/bin/env bash
# run_perf.sh — compare_run.py 的实验封装：统一日志归档 + 环境快照
#
# 日志布局（两级目录，每次运行互不覆盖）：
#   <repo>/kupl_logs/<run-name>/
#     meta.txt          实验档案（git HEAD / 环境变量 / 完整命令——复现三要素）
#     perf.txt          harness 输出（性能表 + 退出码）
#     kuccl_r%r.log     各 rank 的 [KuCCL] 日志（log_write 双写，harness 吞不掉）
#
# 用法：
#   ./run_perf.sh [compare_run.py 原样参数...]
# 环境变量：
#   AG_ALGO      allgather 结构路由：不设=库默认 linear；
#               auto/hier（拓扑合适→分层，否则 linear）/linear/hier_die/hier_socket/smart
#   AG_SYNC      linear 路径同步原语：fence（默认）/doorbell
#   RUN_NAME     二级目录名（默认：月日_时分_algo-X_wsN）
# 示例：
#   ./run_perf.sh --ops all_gather --world-size 16 --trials 10 --maxsize 19 --gate 30
#   RUN_NAME=doorbell AG_SYNC=doorbell ./run_perf.sh --ops all_gather --world-size 16
#
# 原理注释：compare_run.py 用 capture_output=True 吞掉 worker 的 stderr，
# 控制台看不到 [KuCCL] 日志；KUCCL_LOG_FILE 是唯一可靠通道（log_write
# 恒双写 stderr+文件）。文件用追加模式（"a"），加时间戳目录名双保险。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOG_BASE="${REPO_ROOT}/kupl_logs"

AG_ALGO="${AG_ALGO:-}"        # 空=跟随库默认；A/B 时显式设 fence / auto

# 从参数中提取 world-size 供目录命名（不动原参数）
ARGS=("$@")
WS=16
i=0
while [ $i -lt ${#ARGS[@]} ]; do
    if [ "${ARGS[$i]}" = "--world-size" ] && [ $((i + 1)) -lt ${#ARGS[@]} ]; then
        WS="${ARGS[$((i + 1))]}"
    fi
    i=$((i + 1))
done

ALGO_TAG="${AG_ALGO:-default}"
RUN_NAME="${RUN_NAME:-$(date +%m%d_%H%M)_algo-${ALGO_TAG}_ws${WS}}"
RUN_DIR="${LOG_BASE}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

# ── 实验档案：复现所需的三要素（代码版本 / 环境 / 命令）──
{
    echo "git   : $(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "date  : $(date '+%F %T')"
    echo "host  : $(hostname)"
    echo "algo  : ${AG_ALGO:-<library-default>}"
    echo "cmd   : KUCCL_BACKEND_TYPE=kupl KUCCL_KUPL_AG_ALGO=${AG_ALGO:-<unset>} python compare_run.py $*"
    echo "── numa ──"
    lscpu 2>/dev/null | grep -i numa || echo "(lscpu unavailable)"
    echo "── env ──"
    env | grep -E '^KUCCL_' | sort || true
} > "${RUN_DIR}/meta.txt"

export KUCCL_BACKEND_TYPE=kupl
[ -n "${AG_ALGO}" ] && export KUCCL_KUPL_AG_ALGO="${AG_ALGO}"
export KUCCL_KUPL_TIMING="${KUCCL_KUPL_TIMING:-1}"
export KUCCL_LOG_LEVEL="${KUCCL_LOG_LEVEL:-INFO}"
export KUCCL_LOG_FILE="${RUN_DIR}/kuccl_r%r.log"   # 绝对路径：不受 worker CWD 影响

echo "══════════════════════════════════════════════"
echo " run dir : ${RUN_DIR}"
echo " topo    : 看 ${RUN_DIR}/kuccl_r0.log 的 \"topo:\" 行"
echo "══════════════════════════════════════════════"

set +e
python compare_run.py "$@" 2>&1 | tee "${RUN_DIR}/perf.txt"
RC=${PIPESTATUS[0]}
set -e

echo ""
echo "════════ 结果归档（逐项核验存在与大小）════════"
if [ -f "${RUN_DIR}/perf.txt" ]; then
    echo "  性能表  : ${RUN_DIR}/perf.txt ($(du -h "${RUN_DIR}/perf.txt" | cut -f1), exit=${RC})"
else
    echo "  性能表  : ✗ 未生成（harness 输出异常）"
fi
if [ -f "${RUN_DIR}/meta.txt" ]; then
    echo "  实验档案: ${RUN_DIR}/meta.txt ($(du -h "${RUN_DIR}/meta.txt" | cut -f1), 含 git HEAD/NUMA/env)"
else
    echo "  实验档案: ✗ 未生成"
fi
N_LOGS=$(ls "${RUN_DIR}"/kuccl_r*.log 2>/dev/null | wc -l)
if [ "${N_LOGS}" -gt 0 ]; then
    LOG_SIZE=$(du -ch "${RUN_DIR}"/kuccl_r*.log 2>/dev/null | tail -1 | cut -f1)
    echo "  rank日志: ${RUN_DIR}/kuccl_r*.log（${N_LOGS} 个文件, ${LOG_SIZE}）"
else
    echo "  rank日志: ✗ 未生成（检查 KUCCL_LOG_FILE / KUCCL_LOG_LEVEL）"
fi
echo ""
echo "  进入目录: cd ${RUN_DIR}"
echo ""
echo "── 快查 ──"
grep -h "topo:" "${RUN_DIR}/kuccl_r0.log" 2>/dev/null || echo "  (无 topo 行：确认 KUCCL_LOG_LEVEL=INFO)"
grep -h "\[op:" "${RUN_DIR}/kuccl_r0.log" 2>/dev/null | tail -4 || true
exit "${RC}"
