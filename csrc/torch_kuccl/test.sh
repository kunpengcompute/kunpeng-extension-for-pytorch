#!/bin/bash
# test.sh —— KuCCL 一键功能测试（单节点）
#
# 用法：
#   ./test.sh              # 正确性套件（kuccl ws=2 + cpu:kuccl 规格语法 + gloo 反向验证）
#
# 内容：
#   1. tests/correctness.py --world-size 2     kuccl 全算子正确性（26 项）
#   2. tests/correctness.py --backends gloo    用 gloo 跑同一套件（验证测试逻辑本身）
#
# 独有测试（本脚本不跑，按需手动）：
#   python tests/test_subgroup.py             子组/重叠通信域
#   python tests/test_allreduce_coalesced.py  多 tensor 合并归约
#   python tests/test_init_timing.py          建组耗时打点（KuCCL_TIMING=1）
#   python tests/test_spec_case1.py           设计书验收用例 1
#   bash ../benchmarks/communication/bash.md  # DeepSpeed 套件见此速查
#
# 性能 A/B 验证（含环境开关矩阵）见 docs/优化记录.md §4。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 没激活环境就自动 source
if [[ "${CONDA_DEFAULT_ENV:-}" != "${KuCCL_CONDA_ENV:-kuccl291}" ]]; then
    echo "[env] 未激活环境，自动 source ./env.sh ..."
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.sh"
fi

# ── TLS 策略（评审约定）：单机 sm,self（共享内存快路径，不用 cma）──
# 前置：hucx 构建需启用 sm 组件。
# 显式 export 过 UCX_TLS 则尊重用户值。
export UCX_TLS=${UCX_TLS:-sm,self}

echo "══════ [1/3] kuccl 正确性（ws=2，含 stress 背靠背压力）══════"
python "${SCRIPT_DIR}/tests/correctness.py" --world-size 2

echo ""
echo "══════ [2/3] 设计书原语法 cpu:kuccl（设备:后端 前缀格式）══════"
python "${SCRIPT_DIR}/tests/correctness.py" --world-size 2 --backends "cpu:kuccl"

echo ""
echo "══════ [3/3] gloo 反向验证（同一套件，验证测试逻辑本身）══════"
python "${SCRIPT_DIR}/tests/correctness.py" --backends gloo

echo ""
echo "══════ 全部通过 ══════"
echo "下一步（性能 A/B）：docs/优化记录.md §4"
