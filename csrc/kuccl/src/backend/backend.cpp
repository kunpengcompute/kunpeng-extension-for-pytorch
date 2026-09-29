/*
 * backend.cpp —— 后端注册表（名字 → 操作表）
 */

#include "backend.hpp"

#include <cstring>

namespace kuccl {
namespace backend {

const Ops *get(const char *name)
{
    if (name == nullptr || name[0] == '\0')
        return xucg_ops();
    if (std::strcmp(name, "xucg") == 0)
        return xucg_ops();
    if (std::strcmp(name, "ucg") == 0)
        return xucg_ops(); /* 兼容别名 */
    if (std::strcmp(name, "kupl") == 0)
        return kupl_ops();
    return nullptr;
}

} /* namespace backend */
} /* namespace kuccl */
