/*
 * version.cpp —— 库版本查询（生产化设施）
 *
 * 版本唯一事实来源：CMakeLists.txt 的 KUCCL_VERSION 编译定义
 * （与 set_target_properties 的 VERSION/SOVERSION 同步维护）。
 * 运行期经 kucclGetVersion() 读取——出事先问版本，
 * 防"pull 了新代码但跑的是旧 .so"的静默事故。
 */
#include "kuccl.h"

#ifndef KUCCL_VERSION
#define KUCCL_VERSION "unknown"
#endif
#ifndef KUCCL_BUILD_INFO
#define KUCCL_BUILD_INFO "git:unknown built:unknown"
#endif

extern "C" KuCCL_API const char *kucclGetVersion(void)
{
    static const char *ver = "kuccl " KUCCL_VERSION " (" KUCCL_BUILD_INFO "; ABI KuCCL_1.2; backends: xucg, kupl)";
    return ver;
}
