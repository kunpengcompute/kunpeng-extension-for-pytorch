#!/bin/bash
# env.sh —— PyTorch 环境激活（与 kuccl 项目完全独立，任何项目可复用）
#
# 职责只有两件事：
#   1. conda activate（PyTorch 所在环境）
#   2. 探测并 export TORCH_LIB（libtorch_cpu.so 所在目录）
#
# kuccl 的库路径环境（HUCX/XUCG/kupl/LD_LIBRARY_PATH）在 ../kuccl/kuccl_env.sh，
# （本文件位于 torch_kuccl/，两者独立、各自幂等、可单独 source；跑 kuccl 时两个都 source）
#
# 用法：
#   source ./env.sh            # 激活（幂等）
#   source ./env.sh --check    # 激活 + 自检

# PyTorch 所在 conda 环境（中性名；勿用项目前缀——本脚本不属于任何项目）
export PYTORCH_ENV=${PYTORCH_ENV:-kuccl291}

CONDA_SH_CANDIDATES=(
    "/root/pacific_ext/huawei/z00515076/anaconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
)

echo "[env] 激活 PyTorch 环境 ${PYTORCH_ENV} ..."
for INIT in "${CONDA_SH_CANDIDATES[@]}"; do
    if [[ -f "$INIT" ]]; then
        source "$INIT"
        break
    fi
done
if [[ "${CONDA_DEFAULT_ENV}" != "${PYTORCH_ENV}" ]]; then
    conda activate "${PYTORCH_ENV}" 2>/dev/null \
        || { echo "[env] 错误：无法激活 conda 环境 ${PYTORCH_ENV}（conda env list 确认名字）"; return 1 2>/dev/null || exit 1; }
fi

# TORCH_LIB：import torch 探测（首次可能 10-30 秒）
export TORCH_LIB=$(python3 -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent / 'lib')" 2>/dev/null || true)
[[ -z "${TORCH_LIB}" ]] && echo "[env] 警告：TORCH_LIB 探测失败（import torch 异常）"

# torch lib 进链接搜索路径（幂等前置）
case ":${LD_LIBRARY_PATH:-}:" in
    *":${TORCH_LIB}:"*) ;;
    *) export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}" ;;
esac

echo "[env] 完成。python=$(which python 2>/dev/null)  CONDA=${CONDA_DEFAULT_ENV}"

# 自检
if [[ "${1:-}" == "--check" ]]; then
    echo "══════════ PyTorch 环境自检 ══════════"
    [[ "${CONDA_DEFAULT_ENV}" == "${PYTORCH_ENV}" ]]; echo " conda env : ${CONDA_DEFAULT_ENV}  $([[ $? -eq 0 ]] && echo ✓ || echo ✗)"
    [[ -n "${TORCH_LIB}" && -d "${TORCH_LIB}" ]];      echo " TORCH_LIB : ${TORCH_LIB}  $([[ $? -eq 0 ]] && echo ✓ || echo ✗)"
    if python -c "import torch" 2>/dev/null; then
        echo " torch     : ✓ $(python -c 'import torch; print(torch.__version__)')"
    else
        echo " torch     : ✗ import 失败"
    fi
    echo "══════════════════════════"
fi
