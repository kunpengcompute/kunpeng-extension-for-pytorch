#!/bin/bash
# build.sh —— torch_kuccl 适配层一键编译（PyTorch c10d 后端）
#
# 双包独立构建（kuccl 库本体与 torch_kuccl 适配层各自构建）：
#   本目录    —— 适配层（kuccl_backend.so，链接 libkuccl.so，不重编库）
#   ../kuccl/ —— 库本体，有自己的 build.sh
#
# import kuccl 的依赖链（路径约定，勿破坏）：
#   import kuccl
#     → kuccl/__init__.py（ctypes 预加载 HUCX/XUCG 私有库）
#     → kuccl/kuccl_backend*.so（包内，--inplace 产物）
#       → libkuccl.so（../kuccl/build/，RPATH 绝对路径 + env.sh 的 LD_LIBRARY_PATH 双保险）
#
# 用法（在 torch_kuccl/ 目录执行）：
#   ./build.sh            # 增量编译 + import 检查
#   ./build.sh clean      # 清 build/ 与旧 .so 后全量重建
#   ./build.sh fmt        # 代码风格刷（本目录源文件；.clang-format 在仓库根）
#
# 前置：先在 ../kuccl 下完成库构建（./build.sh），否则下方前置检查会明确报错。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$(cd "${SCRIPT_DIR}/../kuccl" && pwd)"   # 库本体（兄弟目录）
MODE="${1:-all}"

# ── 环境：两个独立 env（仓库根 env.sh = PyTorch；kuccl/kuccl_env.sh = 库路径）──
#   shellcheck disable=SC1091
source "${LIB_DIR}/kuccl_env.sh"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${PYTORCH_ENV:-kuccl}" ]]; then
    echo "[env] 未激活 PyTorch 环境，自动 source 仓库根 env.sh ..."
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/../env.sh"
fi

if [[ "${MODE}" == "clean" ]]; then
    echo "── clean：清空 ${SCRIPT_DIR}/build 与旧 .so ..."
    rm -rf "${SCRIPT_DIR}/build"
    rm -f "${SCRIPT_DIR}/kuccl/"kuccl_backend*.so
    MODE="all"
fi

echo "══════════════════════════════════════════════"
echo " conda env : ${CONDA_DEFAULT_ENV:-<未激活>}"
echo " torch     : $(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo '✗')"
echo " libkuccl  : ${LIB_DIR}/build/libkuccl.so"
echo "══════════════════════════════════════════════"

# ── 前置检查（两包独立，缺前置给明确指引而不是裸报错）────────
if ! python -c "import torch" 2>/dev/null; then
    echo "[error] import torch 失败！适配层编译中止。"
    echo "        排查：source ${LIB_DIR}/env.sh --check 看哪项 ✗"
    exit 1
fi
if [[ ! -f "${LIB_DIR}/build/libkuccl.so" ]]; then
    echo "[error] ${LIB_DIR}/build/libkuccl.so 不存在——先构建库本体："
    echo "        cd ${LIB_DIR} && ./build.sh"
    exit 1
fi

build_torch() {
    echo ""
    echo "── 编译 torch_kuccl（适配层；链接 ${LIB_DIR}/build/libkuccl.so）..."
    (
        set -e
        cd "${SCRIPT_DIR}"
        # 无条件全量编译：distutils 只跟踪 .cpp→.o 时间戳、不跟踪头文件，
        # 增量模式下改 include/*.hpp 会被静默跳过（旧 .o 直接链接出新 .so，
        # 头文件改动全部丢失且无报错）。单文件全量一两分钟，买确定性。
        rm -rf build
        rm -f kuccl/kuccl_backend*.so
        CC="${BISHENG_PATH}/bin/clang" CXX="${BISHENG_PATH}/bin/clang++" \
            python setup.py build_ext --inplace
    ) || { echo "[torch_kuccl 失败] 往上翻看编译错误（当前目录未变）"; exit 1; }

    local SO
    SO=$(ls "${SCRIPT_DIR}/kuccl/"kuccl_backend*.so 2>/dev/null | head -1)
    if [[ -z "$SO" ]]; then
        echo "[torch_kuccl 失败] 未生成 kuccl_backend*.so"; exit 1
    fi
    echo "[torch_kuccl 完成] ${SO}"

    # 版本戳：源码 HEAD + 构建时间（benchmark 启动时与 git HEAD 比对防旧二进制）
    {
        git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo "unknown"
        date '+%F %T'
    } > "${SCRIPT_DIR}/kuccl/BUILD_INFO"

    # 检查：import 全链路（ctypes 预加载 UCX/UCG + libkuccl + 注册 backend）
    echo ""
    echo "── 验证 import kuccl ..."
    if (cd "${SCRIPT_DIR}" && python -c "import kuccl; print('OK')"); then
        echo "[检查 通过] import OK —— 依赖链（kuccl 包 → kuccl_backend.so → libkuccl.so）就绪"
    else
        echo "[检查 失败] import 报错。分层排查："
        echo "  1) nm -D ${LIB_DIR}/build/libkuccl.so | grep ' T ' | grep -c kuccl   # 应为 12"
        echo "  2) ldd kuccl_backend*.so | grep kuccl    # 应解析到 ${LIB_DIR}/build 的库"
        exit 1
    fi
}

case "${MODE}" in
    all | "" | torch)
        build_torch
        echo ""
        echo "══════ 适配层构建完成 ══════"
        echo "快速验证:  python tests/correctness.py --world-size 2"
        echo "性能跑分:  python benchmarks/communication/compare_nodes.py --nnodes 8 --ppn 16"
        ;;
    fmt)
        CF_BIN="${CLANG_FORMAT:-${BISHENG_PATH}/bin/clang-format}"
        command -v "${CF_BIN}" >/dev/null 2>&1 || CF_BIN="clang-format"
        echo "── clang-format（${CF_BIN}，torch_kuccl/ 源文件）..."
        git ls-files 'csrc/*.cpp' 'include/*.hpp' \
            | xargs "${CF_BIN}" -i
        echo "完成。git diff 检查后提交。"
        ;;
    *)
        echo "用法: ./build.sh [all|fmt|clean]（默认 all；clean 全量重建）"
        exit 1
        ;;
esac
