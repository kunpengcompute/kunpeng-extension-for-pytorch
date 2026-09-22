#!/bin/bash
# env.sh —— KUCCL 开发环境一键激活
#
# 用法：
#   source ./env.sh            # 激活环境（幂等，可反复 source；每阶段打印进度）
#   source ./env.sh --check    # 激活 + 逐项自检（✓ 才继续干活）
#   source ./env.sh --trace    # set -x 全量跟踪（卡死时精确定位到行）
#
# 流程：conda activate kuccl291 → 设置 HUCX/XUCG/BISHENG/TORCH_LIB/
#       运行库路径（LD_LIBRARY_PATH 显式含 kupl，不依赖 sgl 的 env.sh）
#
# 幂等：重复 source 不会叠加污染 LD_LIBRARY_PATH（自动去重）。

# --trace：set -x 跟踪每一行执行（卡死时最后一行输出 = 卡点）
if [[ "${1:-}" == "--trace" ]]; then
    set -x
fi

# ════════════════════════════════════════════════════════════
# 0. 路径定义（换机器/换目录只改这里）
#    ★ 变量名用 HPCKIT（非 HPCKIT_ROOT）
# ════════════════════════════════════════════════════════════
export HPCKIT=${HPCKIT:-/root/pacific_ext/HPCKit_26.1.RC1_Linux-aarch64/HPCKit/26.1.RC1}
export HPCKIT_ROOT="$HPCKIT"   # 兼容别名

export HUCX_DIR=${HUCX_DIR:-${HPCKIT}/hmpi/bisheng/release/hucx}
export XUCG_DIR=${XUCG_DIR:-${HPCKIT}/hmpi/bisheng/release/xucg}
export BISHENG_PATH=${BISHENG_PATH:-${HPCKIT}/compiler/bisheng}
KUPL_DIR="${KUPL_DIR:-${HPCKIT}/kupl/bisheng/prof}"   # kupl 安装前缀（头+库，kupl shm 后端用）
KUPL_LIB="${KUPL_DIR}/lib"

# ★ 用项目前缀变量名，不用通用的 CONDA_ENV_NAME：外部脚本可能往 shell
#   泄漏 CONDA_ENV_NAME，${CONDA_ENV_NAME:-默认} 会保留泄漏值，激活目标被掉包。
export KUCCL_CONDA_ENV=${KUCCL_CONDA_ENV:-kuccl291}
CONDA_SH_CANDIDATES=(
    "/root/pacific_ext/huawei/z00515076/anaconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
)

# 项目根目录（本脚本所在目录）
export KUCCL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ════════════════════════════════════════════════════════════
# 1. 直接激活工作环境
# ════════════════════════════════════════════════════════════
echo "[env] 1/3 加载 conda 并激活 ${KUCCL_CONDA_ENV} ..."
for INIT in "${CONDA_SH_CANDIDATES[@]}"; do
    if [[ -f "$INIT" ]]; then
        source "$INIT"
        break
    fi
done
if [[ "${CONDA_DEFAULT_ENV}" != "${KUCCL_CONDA_ENV}" ]]; then
    conda activate "${KUCCL_CONDA_ENV}" 2>/dev/null \
        || { echo "[env] 错误：无法激活 conda 环境 ${KUCCL_CONDA_ENV}（conda env list 确认名字）"; return 1 2>/dev/null || exit 1; }
fi
echo "[env]   激活完成：CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"

# ════════════════════════════════════════════════════════════
# 3. 运行库搜索路径（后设，确保优先级最高）
#    ★ TORCH_LIB 必须在内（缺它 import torch 找不到 libtorch_cpu.so）
# ════════════════════════════════════════════════════════════
echo "[env] 2/3 探测 TORCH_LIB（import torch，首次可能 10-30 秒）..."
export TORCH_LIB=$(python3 -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent / 'lib')" 2>/dev/null || true)

# 去重前置：已有路径不重复插入（幂等）
_prepend_unique() {
    local out="" p
    local IFS=':'
    for p in "$@" "${LD_LIBRARY_PATH}"; do
        [[ -z "$p" ]] && continue
        [[ ":${out}:" != *":${p}:"* ]] && out="${out:+${out}:}${p}"
    done
    export LD_LIBRARY_PATH="$out"
}
_prepend_unique "${KUCCL_ROOT}/build" "${KUPL_LIB}" "${BISHENG_PATH}/lib" "${TORCH_LIB}" \
                "${HUCX_DIR}/lib" "${XUCG_DIR}/lib" "${XUCG_DIR}/lib/planc"

# Python 包路径（kuccl 包 + kuccl_backend.so 都在适配层目录）：
# source 本脚本后，任意目录/任意子进程均可 import kuccl——
# 不依赖 pip install（包改名后旧 egg-link 已失效；如需 pip 方式：
#   pip install -e <torch_kuccl目录> --no-build-isolation
#   ★ 必须 --no-build-isolation：pip 默认在隔离"干净房间"里跑 setup.py
#     （房间里只有 setuptools），而 setup.py 要 import torch）
TORCH_KUCCL_DIR="$(cd "${KUCCL_ROOT}/.." && pwd)/torch_kuccl"
export PYTHONPATH="${TORCH_KUCCL_DIR}:${PYTHONPATH:-}"

# UCX 动态模块目录（限定加载我们自己这套，防串到系统 UCX）
export UCX_MODULE_DIR="${HUCX_DIR}/lib/ucx"

# ════════════════════════════════════════════════════════════
# 4. 运行参数（默认值，按需覆盖；export 过的值优先）
# ════════════════════════════════════════════════════════════
# TLS 策略：跨节点 rc（不绑网卡——单卡绑定的设备名在部分节点解析失败）
export UCX_TLS=${UCX_TLS:-rc}
export UCG_PLANC_UCX_BCAST_ATTR=${UCG_PLANC_UCX_BCAST_ATTR:-I:1}

# ★ 线程池约束：OMP 默认用满全核，线程栅栏成本会污染通信延迟
#   （256K 档实测差 10 倍）：
#   单机测试/基准默认 1 线程（torchrun 同款口径）；
#   多进程训练部署请覆盖为 核数/PPN。显式设置过则尊重用户值。
# export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
# export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
# export KUCCL_TIMING=1                              # 需要耗时打点时取消注释
# export KUCCL_INIT_DEBUG=1                          # 需要建组调试日志时取消注释
# export KUCCL_OOB_DEBUG=1                           # 需要 OOB 调试日志时取消注释

unset MODULES_RUN_QUARANTINE 2>/dev/null || true

echo "[env] 3/3 完成。python=$(which python 2>/dev/null)"
[[ -z "${TORCH_LIB}" ]] && echo "[env] 警告：TORCH_LIB 探测失败（import torch 异常）——用 --check 排查"

# ════════════════════════════════════════════════════════════
# 5. 自检（--check 参数）
# ════════════════════════════════════════════════════════════
if [[ "${1:-}" == "--check" ]]; then
    echo "══════════ KUCCL 环境自检 ══════════"
    _ok() { [[ "$1" == "0" ]] && echo "✓" || echo "✗"; }

    [[ "${CONDA_DEFAULT_ENV}" == "${KUCCL_CONDA_ENV}" ]]; echo " conda env   : ${CONDA_DEFAULT_ENV}  $(_ok $?)"
    echo " python      : $(which python)"
    [[ -d "${HUCX_DIR}" ]];                              echo " HUCX_DIR    : ${HUCX_DIR}  $(_ok $?)"
    [[ -d "${XUCG_DIR}" ]];                              echo " XUCG_DIR    : ${XUCG_DIR}  $(_ok $?)"
    [[ -x "${BISHENG_PATH}/bin/clang++" ]];              echo " BISHENG     : ${BISHENG_PATH}  $(_ok $?)"
    [[ -d "${KUPL_LIB}" ]];                              echo " KUPL lib    : ${KUPL_LIB}  $(_ok $?)"
    [[ -n "${TORCH_LIB}" && -d "${TORCH_LIB}" ]];        echo " TORCH_LIB   : ${TORCH_LIB}  $(_ok $?)"
    [[ -f "${HUCX_DIR}/lib/libucp.so" ]];                echo " libucp.so   : $(_ok $?)"
    [[ -f "${XUCG_DIR}/lib/libucg.so" ]];                echo " libucg.so   : $(_ok $?)"
    echo " KUCCL_ROOT  : ${KUCCL_ROOT}"
    echo "─────────────────────────────"
    if python -c "import torch" 2>/dev/null; then
        echo " torch       : ✓ $(python -c 'import torch; print(torch.__version__)')"
    else
        echo " torch       : ✗ import 失败！检查 LD_LIBRARY_PATH 是否含 kupl/TORCH_LIB"
    fi
    if [[ -f "${KUCCL_ROOT}/build/libkuccl.so" ]]; then
        echo " libkuccl.so : ✓ 已编译"
    else
        echo " libkuccl.so : 未编译（./build.sh）"
    fi
    if ls "${KUCCL_ROOT}/../torch_kuccl"/kuccl_backend*.so >/dev/null 2>&1; then
        echo " 适配层 .so  : ✓ 已编译"
    else
        echo " 适配层 .so  : 未编译（../torch_kuccl 下 ./build.sh）"
    fi
    echo "════════════════════════════"
fi
