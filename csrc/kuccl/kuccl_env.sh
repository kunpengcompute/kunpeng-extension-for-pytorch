#!/bin/bash
# kuccl_env.sh —— KuCCL 库运行环境（只管库路径，不管 PyTorch）
#
# 与仓库根 env.sh（PyTorch 环境激活）完全独立：
#   根 env.sh            → conda activate + TORCH_LIB（任何项目可复用）
#   本脚本（kuccl_env.sh）→ HPCKit/HUCX/XUCG/kupl/BISHENG 库路径 + UCX 默认值
#
# 用法：source ./kuccl_env.sh（幂等，可反复 source；不激活任何 conda 环境）
#
# 纯库构建（build.sh）只 source 本脚本即可——kuccl 库本体零 torch 依赖；
# 跑 torch_kuccl / benchmark 时两个都 source。

# ── 路径定义（换机器/换目录只改这里）─────────────────────────
export HPCKIT=${HPCKIT:-/root/pacific_ext/HPCKit_26.2.0/HPCKit/26.2.0}
export HPCKIT_ROOT="$HPCKIT"   # 兼容别名

export HUCX_DIR=${HUCX_DIR:-${HPCKIT}/hmpi/bisheng/release/hucx}
export XUCG_DIR=${XUCG_DIR:-${HPCKIT}/hmpi/bisheng/release/xucg}
export BISHENG_PATH=${BISHENG_PATH:-${HPCKIT}/compiler/bisheng}
KUPL_DIR="${KUPL_DIR:-${HPCKIT}/kupl/bisheng/prof}"   # kupl 安装前缀（头+库）
KUPL_LIB="${KUPL_DIR}/lib"

# 项目根目录（本脚本所在目录）
export KuCCL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 库搜索路径（去重前置，幂等）─────────────────────────────
_prepend_unique() {
    local out="" p
    local IFS=':'
    for p in "$@" "${LD_LIBRARY_PATH}"; do
        [[ -z "$p" ]] && continue
        [[ ":${out}:" != *":${p}:"* ]] && out="${out:+${out}:}${p}"
    done
    export LD_LIBRARY_PATH="$out"
}
_prepend_unique "${KuCCL_ROOT}/build" "${KUPL_LIB}" "${BISHENG_PATH}/lib" \
                "${HUCX_DIR}/lib" "${XUCG_DIR}/lib" "${XUCG_DIR}/lib/planc"

# Python 包路径（kuccl 包 + kuccl_backend.so 在适配层目录）：
# source 本脚本后任意目录/子进程可 import kuccl（不依赖 pip install；
# 如需 pip 方式：pip install -e <torch_kuccl目录> --no-build-isolation
#   ★ 必须 --no-build-isolation：setup.py 要 import torch，隔离房间没有）
TORCH_KuCCL_DIR="$(cd "${KuCCL_ROOT}/.." && pwd)/torch_kuccl"
export PYTHONPATH="${TORCH_KuCCL_DIR}:${PYTHONPATH:-}"

# UCX 动态模块目录（限定加载我们自己这套，防串到系统 UCX）
export UCX_MODULE_DIR="${HUCX_DIR}/lib/ucx"

# ── 运行参数（默认值；export 过的值优先）─────────────────────
# TLS 策略：rc 跨节点 + sm 楼内（同节点对自动选 sm，node-aware 算法的
# 楼内阶段依赖 shm）；不绑网卡——单卡绑定的设备名在部分节点解析失败
export UCX_TLS=${UCX_TLS:-rc,sm,self}
# bcast 算法锁定：I:1 = 全尺寸强制 1 号扁平 Binomial tree。
# 拓扑喂入后 8_16 表接管更优，默认注释；bcast 回退时取消注释：
# export UCG_PLANC_UCX_BCAST_ATTR=${UCG_PLANC_UCX_BCAST_ATTR:-I:1}

unset MODULES_RUN_QUARANTINE 2>/dev/null || true

echo "[kuccl_env] 库路径就绪。HUCX=${HUCX_DIR}  XUCG=${XUCG_DIR}"

# 自检（--check）
if [[ "${1:-}" == "--check" ]]; then
    echo "══════════ KuCCL 库环境自检 ══════════"
    _ok() { [[ "$1" == "0" ]] && echo "✓" || echo "✗"; }
    [[ -d "${HUCX_DIR}" ]];                              echo " HUCX_DIR    : ${HUCX_DIR}  $(_ok $?)"
    [[ -d "${XUCG_DIR}" ]];                              echo " XUCG_DIR    : ${XUCG_DIR}  $(_ok $?)"
    [[ -x "${BISHENG_PATH}/bin/clang++" ]];              echo " BISHENG     : ${BISHENG_PATH}  $(_ok $?)"
    [[ -d "${KUPL_LIB}" ]];                              echo " KUPL lib    : ${KUPL_LIB}  $(_ok $?)"
    [[ -f "${HUCX_DIR}/lib/libucp.so" ]];                echo " libucp.so   : $(_ok $?)"
    [[ -f "${XUCG_DIR}/lib/libucg.so" ]];                echo " libucg.so   : $(_ok $?)"
    echo " KuCCL_ROOT  : ${KuCCL_ROOT}"
    echo "─────────────────────────────"
    if [[ -f "${KuCCL_ROOT}/build/libkuccl.so" ]]; then
        echo " libkuccl.so : ✓ 已编译"
    else
        echo " libkuccl.so : 未编译（./build.sh）"
    fi
    if ls "${KuCCL_ROOT}/../torch_kuccl"/kuccl/kuccl_backend*.so >/dev/null 2>&1; then
        echo " 适配层 .so  : ✓ 已编译"
    else
        echo " 适配层 .so  : 未编译（../torch_kuccl 下 ./build.sh）"
    fi
    echo "══════════════════════════"
fi
