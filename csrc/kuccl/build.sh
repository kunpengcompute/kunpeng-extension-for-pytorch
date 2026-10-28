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
# 检查：符号表与 kuccl.map 逐个核对（12 公开接口，version script 双保险）
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
        # ★ HEAD 变化时强制 re-configure：git hash 在 configure 期经
        #   execute_process 取得（内嵌进 .so 的构建戳）——只跑 make 不
        #   重跑 configure 的话，hash 停留在旧值，"防旧库"闭环失效
        CUR_HEAD=$(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)
        if [[ ! -f Makefile || "$(cat BUILD_STAMP 2>/dev/null)" != "${CUR_HEAD}" ]]; then
            echo "── cmake configure（刷新构建戳 git:${CUR_HEAD}）..."
            CC="${BISHENG_PATH}/bin/clang" CXX="${BISHENG_PATH}/bin/clang++" \
                cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                -DKUCCL_GIT_HEAD_EXTERN="${CUR_HEAD}"
            echo "${CUR_HEAD}" > BUILD_STAMP
            # ★ HEAD 变化 → 强制重链：删除旧 .so。防两类事故：
            #   1) 中断的链接留下残缺 .so，mtime 新于所有源文件，
            #      增量 make 永远视其为"最新"（实测：符号表可读但加载失败）
            #   2) pull 后 .so 与源码版本错位（防旧库闭环的链接侧补强）
            rm -f "${SCRIPT_DIR}"/build/libkuccl.so*
            # 时钟偏移防御：git 戳在 version.cpp 编译期内嵌；集群文件系统
            # 常见 ~几十 ms 的"未来时间戳"（make 警告 Clock skew detected），
            # 会让 make 误判 version.cpp.o 比依赖新而跳过重编 → 戳校验误报
            # 失败 exit 1 → 上层 && 链断裂。configure 刷新时强制删其 .o，
            # 单文件重编，成本为零，换取戳必然更新。
            rm -f CMakeFiles/kuccl.dir/src/version.cpp.o
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

    # 检查①：动态符号表与 kuccl.map 逐个核对（version script 双保险的核对侧）
    local MAP="${SCRIPT_DIR}/kuccl.map"
    local exported n miss
    exported=$(nm -D "$LIB" | grep ' T ' | grep kuccl || true)
    n=$(echo "$exported" | grep -c kuccl || true)
    if [[ "$n" -ne 13 ]]; then
        echo "[检查失败] libkuccl.so 导出 ${n}/13 个 kuccl 符号"
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
        echo "[检查失败] map 清单中缺失符号（对应实现漏 KuCCL_API 或未编译）:$miss"
        exit 1
    fi
    echo "[检查 通过] 13 个公开接口全部导出且与 kuccl.map 一致"

    # ★★ 防旧库闭环校验（pipefail 安全版）：
    #    历史坑：本脚本 set -euo pipefail，而 `strings | grep -q` 里
    #    grep -q 命中即退 → strings（输出远超管道缓冲）收到 SIGPIPE
    #    退出码 141 → pipefail 判整条管道失败——字符串明明在 .so 里，
    #    校验却报失配（clean 重建也复现，且随命中位置时好时坏）。
    #    修复：先捕获 strings 输出，再用 bash 子串判定——无管道、无早退。
    local CUR_HEAD EMB
    CUR_HEAD=$(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)
    EMB=$(strings "$REAL" 2>/dev/null)
    if [[ "$CUR_HEAD" != "unknown" ]] && [[ "$EMB" != *"git:${CUR_HEAD} "* ]]; then
        echo "[构建戳] 首查失配——自愈：强制重编 version.cpp 后复查 ..."
        (cd "${SCRIPT_DIR}/build" && rm -f CMakeFiles/kuccl.dir/src/version.cpp.o && make -j"$(nproc)")
        REAL=$(readlink -f "$LIB")
        EMB=$(strings "$REAL" 2>/dev/null)
    fi
    if [[ "$CUR_HEAD" != "unknown" ]] && [[ "$EMB" == *"git:${CUR_HEAD} "* ]]; then
        echo "[构建戳] ✓ libkuccl.so 内嵌 git:${CUR_HEAD}（运行首行 [KuCCL] init 日志可见同一戳）"
    else
        echo "[构建戳] ★★★ 自愈后仍失配——三处证据定位断点："
        echo "  源码 HEAD       : ${CUR_HEAD}"
        echo "  flags.make 定义 : $(grep -o 'git:[a-f0-9]* built:[0-9: -]*' "${SCRIPT_DIR}/build/CMakeFiles/kuccl.dir/flags.make" 2>/dev/null | head -1)"
        echo "  .so 实际嵌入    : $(strings "$REAL" 2>/dev/null | grep -o 'git:[a-f0-9]* built:[0-9: -]*' | head -1)"
        echo "  判读： HEAD≠flags.make → configure 未刷新（查 BUILD_STAMP）；"
        echo "        flags.make≠.so → version.cpp.o 陈旧（重编仍未生效，查时间戳）"
        echo "  强制全量重建：./build.sh clean && ./build.sh"
        exit 1
    fi

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
        echo "完成。git diff 检查后提交；重跑 ./build.sh 确认编译与符号检查。"
        ;;
    clean) ;;  # 已在上面处理
    *)
        echo "用法: ./build.sh [lib|fmt|clean]（默认 lib 增量；clean 全量重建）"
        exit 1
        ;;
esac
