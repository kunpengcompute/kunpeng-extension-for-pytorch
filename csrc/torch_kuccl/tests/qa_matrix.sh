#!/usr/bin/env bash
# qa_matrix.sh —— kupl 后端回归矩阵：world_size × 重复轮次
#
# 用法（先建环境再跑）：
#   cd csrc/kuccl && source ./kuccl_env.sh && ./build.sh
#   cd ../torch_kuccl && ./build.sh
#   ./tests/qa_matrix.sh
#
# 可调参数（环境变量）：
#   WORLD_SIZES  参与矩阵的 world_size 列表（默认 "2 4 8 16"）
#   ROUNDS       每个 ws 重复轮数（默认 3；竞态要靠重复逼出来）
#   ITERS        传给 correctness 的 --iters（默认 20；长稳可设 200）
#   TIMEOUT_S    单次运行超时秒数（默认 1200；挂死也算 FAIL）
#
# 判定标准（缺一不可）：
#   退出码 0，且输出含 "ALL PASSED"
#
# 退出码 0 = 全部通过；非 0 = 有失败。
# 每轮完整日志保留在 qa_ws<ws>_r<round>.log 供排查。

set -u
cd "$(dirname "$0")/.."   # torch_kuccl 根目录

WORLD_SIZES=${WORLD_SIZES:-"2 4 8 16"}
ROUNDS=${ROUNDS:-3}
ITERS=${ITERS:-20}
TIMEOUT_S=${TIMEOUT_S:-1200}
BACKEND=${BACKEND:-kupl}

export KUCCL_BACKEND_TYPE=$BACKEND
echo "=== kupl 回归矩阵: ws=[$WORLD_SIZES] × ${ROUNDS}轮, iters=${ITERS}, 超时${TIMEOUT_S}s ==="

pass=0
fail=0
failed_runs=""

for ws in $WORLD_SIZES; do
    for r in $(seq 1 "$ROUNDS"); do
        tag="ws=${ws} round=${r}"
        log="qa_ws${ws}_r${r}.log"
        printf '▶ %-18s ' "$tag"

        if timeout "$TIMEOUT_S" python tests/correctness.py \
               --world-size "$ws" --iters "$ITERS" --timeout 900 > "$log" 2>&1; then
            # ★ 判定看全文的 RESULT 行：进程退出噪音（SDMA/TCPStore 警告）
            #   可能打印在 RESULT 之后，tail 窗口判定会被挤出而误判
            if grep -q "^\[RESULT\] ALL PASSED" "$log"; then
                echo "PASS"
                pass=$((pass + 1))
                rm -f "$log"   # 干净通过不留日志
            else
                echo "FAIL（退出码 0 但结果非 ALL PASSED）"
                grep -E "^\[RESULT\]|^  ✗|FAIL " "$log" | head -10
                fail=$((fail + 1))
                failed_runs="$failed_runs [$tag]"
            fi
        else
            rc=$?
            if [ "$rc" -eq 124 ]; then
                echo "FAIL（挂死，超时 ${TIMEOUT_S}s）"
            else
                echo "FAIL（退出码 $rc）"
            fi
            tail -5 "$log"
            fail=$((fail + 1))
            failed_runs="$failed_runs [$tag]"
        fi
    done
done

echo "==============================================="
echo "结果: PASS=${pass}  FAIL=${fail}  (backend=${BACKEND})"
if [ -n "$failed_runs" ]; then
    echo "失败轮次:${failed_runs}"
    echo "失败日志: ./qa_ws*_r*.log"
    grep -h "^  ✗\|FAIL" qa_ws*_r*.log 2>/dev/null | sort -u | head -20
fi

[ "$fail" -eq 0 ] || exit 1
exit 0
