# kuccl — 鲲鹏集合通信库（libkuccl.so）
#
# 独立构建，零框架依赖。只链接 UCX/UCG。
#
# 构建：
#   export HUCX_DIR=/path/to/hucx
#   export XUCG_DIR=/path/to/xucg
#   mkdir build && cd build
#   cmake .. && make -j
#
# 产物：libkuccl.so

cmake_minimum_required(VERSION 3.18)
project(kuccl LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# 安装前缀（来自环境变量）
if(DEFINED ENV{HUCX_DIR})
    set(HUCX_DIR "$ENV{HUCX_DIR}" CACHE PATH "HUCX (UCX) install prefix")
endif()
if(DEFINED ENV{XUCG_DIR})
    set(XUCG_DIR "$ENV{XUCG_DIR}" CACHE PATH "XUCG (UCG) install prefix")
endif()
if(NOT HUCX_DIR)
    message(FATAL_ERROR "HUCX_DIR 未设置。请 export HUCX_DIR=/path/to/hucx")
endif()
if(NOT XUCG_DIR)
    message(FATAL_ERROR "XUCG_DIR 未设置。请 export XUCG_DIR=/path/to/xucg")
endif()

find_library(UCP_LIB ucp HINTS "${HUCX_DIR}/lib" "${HUCX_DIR}/lib64" REQUIRED)
find_library(UCT_LIB uct HINTS "${HUCX_DIR}/lib" "${HUCX_DIR}/lib64" REQUIRED)
find_library(UCS_LIB ucs HINTS "${HUCX_DIR}/lib" "${HUCX_DIR}/lib64" REQUIRED)
find_library(UCM_LIB ucm HINTS "${HUCX_DIR}/lib" "${HUCX_DIR}/lib64" REQUIRED)
find_library(UCG_LIB ucg HINTS "${XUCG_DIR}/lib" "${XUCG_DIR}/lib64" REQUIRED)

# ── KUPL（可选）：设 KUPL_DIR 指向 kupl 安装前缀则启用 kupl shm 后端，
#    否则 kupl.cpp 编为存根（选择 kupl 后端时报"未链接"错误）──
if(DEFINED ENV{KUPL_DIR})
    set(KUPL_DIR "$ENV{KUPL_DIR}" CACHE PATH "KUPL install prefix")
endif()
set(KuCCL_HAVE_KUPL OFF)
if(KUPL_DIR)
    find_path(KUPL_INCLUDE_DIR kupl.h HINTS "${KUPL_DIR}/include")
    find_library(KUPL_LIBRARY kupl HINTS "${KUPL_DIR}/lib" "${KUPL_DIR}/lib64")
    if(KUPL_INCLUDE_DIR AND KUPL_LIBRARY)
        set(KuCCL_HAVE_KUPL ON)
    else()
        message(STATUS "kupl shm 后端: KUPL_DIR 下未找到 kupl.h/libkupl，编为存根")
    endif()
else()
    message(STATUS "kupl shm 后端: 未设 KUPL_DIR，编为存根")
endif()

# ── libkuccl.so：纯库，不含任何 torch 代码 ──
add_library(kuccl SHARED
    src/comm.cpp
    src/coll.cpp
    src/error.cpp
    src/lock.cpp
    src/log.cpp
    src/backend/backend.cpp
    src/backend/xucg.cpp
    src/backend/kupl.cpp
)

target_include_directories(kuccl PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"    # 用户 #include <kuccl.h>
)

target_include_directories(kuccl PRIVATE
    "${HUCX_DIR}/include"
    "${XUCG_DIR}/include"
)

target_link_libraries(kuccl PRIVATE
    ${UCP_LIB} ${UCT_LIB} ${UCS_LIB} ${UCM_LIB} ${UCG_LIB}
)

if(KuCCL_HAVE_KUPL)
    target_compile_definitions(kuccl PRIVATE KuCCL_HAVE_KUPL=1)
    target_include_directories(kuccl PRIVATE "${KUPL_INCLUDE_DIR}")
    target_link_libraries(kuccl PRIVATE ${KUPL_LIBRARY})
    message(STATUS "kupl shm 后端: 启用 (${KUPL_LIBRARY})")
endif()

# ── 导出清单双保险（version script）─────────────────────────
# 编译期：-fvisibility=hidden + KuCCL_API 宏（单元级符号控制）
# 链接期：kuccl.map 强制最终导出表 = 13 符号（12 公开 + AllToAllV 内部）
# 双保险：漏宏/漏 map 任一 → 动态符号表缺失 → build.sh 门禁① 构建期拦截
target_link_options(kuccl PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/kuccl.map")

target_compile_options(kuccl PRIVATE -O2 -fvisibility=hidden)
# 头文件依赖跟踪：改 src/*.hpp（如 comm_impl.hpp）也触发对应 .cpp 重编
# （make 默认只看 .cpp 时间戳；GCC/Clang 的 -MMD -MP 生成 .d 依赖文件）
target_compile_options(kuccl PRIVATE -MMD -MP)
# KuCCL_EXPORTS：让 kuccl.h 里的 KuCCL_API 宏生效，
# 公开接口符号可见（其余内部符号保持 hidden）
target_compile_definitions(kuccl PRIVATE KuCCL_EXPORTS=1)

set_target_properties(kuccl PROPERTIES
    LIBRARY_OUTPUT_NAME kuccl                # libkuccl.so
    BUILD_RPATH   "${HUCX_DIR}/lib;${XUCG_DIR}/lib"
    INSTALL_RPATH "${HUCX_DIR}/lib;${XUCG_DIR}/lib"
    VERSION 2.6
    SOVERSION 2
)

install(TARGETS kuccl LIBRARY DESTINATION lib)
install(FILES
    include/kuccl.h
    include/types.hpp
    DESTINATION include)

message(STATUS "kuccl: 纯库零框架依赖（xucg/kupl 双后端）")
