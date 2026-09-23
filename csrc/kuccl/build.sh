#!/bin/bash
# build.sh —— kuccl 库本体一键编译（纯库，零 torch 依赖）
#
# 双包独立构建（kuccl 库本体与 torch_kuccl 适配层各自构建）：
#   kuccl/       本目录 —— 库本体（libkuccl.so）
#   torch_kuccl/ 兄弟目录 —— PyTorch 适配层，有自己的 build.sh（链接本库，不重编）
#
# 用法（在 kuccl/ 目录执行）：
#   ./build.sh            # ★ 日常用这个：增量编译（改了什么就编什么）
#   ./build.sh clean      # 全量重建（清 build/ 后完整编）
#   ./build.sh fmt        # 代码风格刷（本目录源文件；.clang-format 在仓库根）
#
# 门禁：符号表与 kuccl.map 逐个核对（12 公开接口，version script 双保险）
#
# 环境依赖：自动 source ./kuccl_env.sh（HUCX/XUCG/BISHENG 库路径）。
# 纯库构建零 torch/conda 依赖——PyTorch 环境由仓库根 env.sh 管，与本构建无关。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-lib}"

# ── 自动激活库路径环境（幂等，不含 conda）────────────────────
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/kuccl_env.sh"

# clean 模式：清构建目录后走完整构建
if [[ "${MODE}" == "clean" ]]; then
    echo "── clean：清空 ${SCRIPT_DIR}/build ..."
    rm -rf "${SCRIPT_DIR}/build"
    MODE="lib"
fi

echo "══════════════════════════════════════════════"
echo " conda env : ${CONDA_DEFAULT_ENV:-<纯库构建不需要>}"
echo " HUCX_DIR  = ${HUCX_DIR}"
echo " XUCG_DIR  = ${XUCG_DIR}"
echo " BISHENG   = ${BISHENG_PATH}"
echo "══════════════════════════════════════════════"

build_lib() {
    echo ""
    echo "── 编译 kuccl 库（make 增量：只编改动的）..."
    (
        set -e
        cd "${SCRIPT_DIR}"
        # ★ 不 rm -rf build：保留 cmake 缓存与 .o 文件
        #   make 按依赖跟踪只重编改动过的 .cpp（改 src/*.hpp 也会触发：
        #   CMakeLists 已配 .d 依赖文件）
        mkdir -p build && cd build
        if [[ ! -f Makefile ]]; then     # 首次（或 clean 后）才跑 cmake configure
            CC="${BISHENG_PATH}/bin/clang" CXX="${BISHENG_PATH}/bin/clang++" \
                cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
        fi
        make -j"$(nproc)"
    ) || { echo "[kuccl 库失败] 往上翻看编译错误（当前目录未变）"; exit 1; }

    local LIB="${SCRIPT_DIR}/build/libkuccl.so"
    if [[ ! -f "$LIB" ]]; then
        echo "[kuccl 库失败] 未生成 ${LIB}"; exit 1
    fi
    # SONAME 化后 $LIB 是软链——显示真实产物和大小
    local REAL
    REAL=$(readlink -f "$LIB")
    echo "[kuccl 库完成] $REAL ($(stat -c %s "$REAL") bytes)"

    # 门禁①：动态符号表与 kuccl.map 逐个核对（version script 双保险的核对侧）
    local MAP="${SCRIPT_DIR}/kuccl.map"
    local exported n miss
    exported=$(nm -D "$LIB" | grep ' T ' | grep kuccl || true)
    n=$(echo "$exported" | grep -c kuccl || true)
    if [[ "$n" -ne 12 ]]; then
        echo "[门禁失败] libkuccl.so 导出 ${n}/12 个 kuccl 符号"
        echo "          排查：nm -D $LIB | grep ' T ' | grep kuccl"
        exit 1
    fi
    # 逐个核对 map 清单。
    # ★ version script 生效后符号带版本后缀（kucclAllGather@@KuCCL_1.1），
    #   匹配模式：符号名后跟 '@'（版本）或行尾——锚定行尾会漏配带版本符号
    miss=""
    while read -r sym; do
        [[ -z "$sym" ]] && continue
        echo "$exported" | grep -qE " ${sym}(@|$)" || miss="$miss $sym"
    done < <(grep -E '^\s+kuccl' "$MAP" | tr -d ' ;')
    if [[ -n "$miss" ]]; then
        echo "[门禁失败] map 清单中缺失符号（对应实现漏 KuCCL_API 或未编译）:$miss"
        exit 1
    fi
    echo "[门禁 通过] 12 个公开接口全部导出且与 kuccl.map 一致"

    # 版本戳：源码 HEAD + 构建时间。benchmark 启动时与当前 git HEAD 比对，
    # 不一致即告警——防"pull 了新代码但跑的是旧 .so"的静默事故
    {
        git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo "unknown"
        date '+%F %T'
    } > "${SCRIPT_DIR}/build/BUILD_INFO"
}

case "${MODE}" in
    lib | all | "")
        build_lib
        echo ""
        echo "══════ 库构建完成 ══════"
        echo "适配层构建：cd ../torch_kuccl && ./build.sh"
        ;;
    fmt)
        # 代码风格统一刷（.clang-format 在仓库根；优先毕昇自带）
        CF_BIN="${CLANG_FORMAT:-${BISHENG_PATH}/bin/clang-format}"
        command -v "${CF_BIN}" >/dev/null 2>&1 || CF_BIN="clang-format"
        echo "── clang-format（${CF_BIN}，kuccl/ 源文件）..."
        # ★ pathspec 相对本目录（git ls-files 从 CWD 解析）
        git ls-files 'src/*.cpp' 'src/*.hpp' 'src/backend/*.cpp' 'src/backend/*.hpp' \
                     'include/*.h' 'include/*.hpp' \
            | xargs "${CF_BIN}" -i
        echo "完成。git diff 检查后提交；重跑 ./build.sh 确认编译与门禁。"
        ;;
    clean) ;;  # 已在上面处理
    *)
        echo "用法: ./build.sh [lib|fmt|clean]（默认 lib 增量；clean 全量重建）"
        exit 1
        ;;
esac
