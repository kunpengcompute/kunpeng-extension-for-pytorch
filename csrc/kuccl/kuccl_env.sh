#!/bin/bash
# kuccl_env.sh —— KuCCL 库构建与运行环境
#
# ★ 唯一需要配置的变量：HPCKIT（HPCKit 安装根目录）
#   其他所有路径（HUCX/XUCG/KUPL/BISHENG/HMPI）都自动从它推导。
#
# 用法：
#   source ./kuccl_env.sh              # 直接用（默认路径）
#   export HPCKIT=/your/path           # 换机器时改这一个就行
#   source ./kuccl_env.sh
#   source ./kuccl_env.sh --check     # 自检

# ════════════════════════════════════════════════════════════
# ★ 唯一配置项：HPCKit 安装根目录（换机器只改这里）
# ════════════════════════════════════════════════════════════
export HPCKIT=${HPCKIT:-/root/pacific_ext/HPCKit_26.2.0/HPCKit/26.2.0}

# ═══ 以下全部自动推导，勿改 ════════════════════════════════
export HUCX_DIR=${HUCX_DIR:-${HPCKIT}/hmpi/bisheng/release/hucx}
export XUCG_DIR=${XUCG_DIR:-${HPCKIT}/hmpi/bisheng/release/xucg}
export KUPL_DIR=${KUPL_DIR:-${HPCKIT}/kupl/bisheng/prof}
export BISHENG_PATH=${BISHENG_PATH:-${HPCKIT}/compiler/bisheng}
export HMPI=${HMPI:-${HPCKIT}/hmpi/bisheng/release/hmpi}

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

# hmpi/lib goes LAST in priority: hmpi may bundle its own UCX/UCG copies
# that must not shadow the HUCX/XUCG libs kuccl was linked against.
_prepend_unique "${HUCX_DIR}/lib" "${XUCG_DIR}/lib" "${XUCG_DIR}/lib/planc" \
                "${KUPL_DIR}/lib" "${BISHENG_PATH}/lib" "${HMPI}/lib" \
                "${KuCCL_ROOT}/build"

# UCX 动态模块目录（限定加载我们自己这套，防串到系统 UCX）
export UCX_MODULE_DIR="${HUCX_DIR}/lib/ucx"

# ── 编译器路径 ─────────────────────────────────────────────
export PATH="${BISHENG_PATH}/bin:${HMPI}/bin:${PATH}"
export OPAL_PREFIX="${HMPI}"

# 编译器（pip install / cmake / make 等构建工具读这俩变量）
export CC=${BISHENG_PATH}/bin/clang
export CXX=${BISHENG_PATH}/bin/clang++

# ── UCX 运行参数（默认值，按需覆盖）───────────────────────
export UCX_TLS=${UCX_TLS:-sm,self}

echo "[kuccl_env] HPCKIT=${HPCKIT}"
echo "[kuccl_env] HUCX=${HUCX_DIR}"
echo "[kuccl_env] XUCG=${XUCG_DIR}"
echo "[kuccl_env] KUPL=${KUPL_DIR}"
echo "[kuccl_env] BISHENG=${BISHENG_PATH}"

# ── 自检模式 ──────────────────────────────────────────────
if [[ "${1:-}" == "--check" ]]; then
    echo ""
    echo "══════════ kuccl_env 自检 ══════════"
    _ok() { [[ $? -eq 0 ]] && echo "✓" || echo "✗"; }
    [[ -d "${HPCKIT}" ]];                            echo " HPCKIT     : ${HPCKIT}  $(_ok)"
    [[ -d "${HUCX_DIR}/lib" ]];                      echo " HUCX lib   : ${HUCX_DIR}  $(_ok)"
    [[ -d "${XUCG_DIR}/lib" ]];                      echo " XUCG lib   : ${XUCG_DIR}  $(_ok)"
    [[ -d "${KUPL_DIR}/lib" ]];                      echo " KUPL lib   : ${KUPL_DIR}  $(_ok)"
    [[ -d "${BISHENG_PATH}/bin" ]];                  echo " BISHENG    : ${BISHENG_PATH}  $(_ok)"
    [[ -x "${BISHENG_PATH}/bin/clang++" ]];           echo " clang++    : $(${BISHENG_PATH}/bin/clang++ --version 2>/dev/null | head -1)  $(_ok)"
    [[ -f "${HUCX_DIR}/lib/libucp.so" ]];             echo " libucp.so  : $(_ok)"
    [[ -f "${XUCG_DIR}/lib/libucg.so" ]];             echo " libucg.so  : $(_ok)"
    echo "══════════════════════════"
fi
