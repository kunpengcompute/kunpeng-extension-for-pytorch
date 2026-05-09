/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * KPEX is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *        http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef KPEX_UTILS_CHECK_H
#define KPEX_UTILS_CHECK_H

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace kpex {
namespace internal {
inline void check_fail_print(std::stringstream &stream) {}

template <typename Arg, typename... Rest>
inline void check_fail_print(std::stringstream &stream, Arg &&arg, Rest &&...rest)
{
    stream << std::forward<Arg>(arg);
    check_fail_print(stream, rest...);
}

template <typename... Args>
inline void check_fail(std::string func, std::string file, int line, Args &&...args)
{
    std::stringstream stream;
    stream << "KPEX_CHECK fail in " << func << " at " << file << ":" << line << ", ";
    check_fail_print(stream, std::forward<Args>(args)...);
    stream << "\n";
    std::cerr << stream.str();
    abort();
}
} //namespace internal
} //namespace kpex

#define KPEX_CHECK(condition, ...)                                                 \
    do {                                                                           \
        if (__builtin_expect(!(condition), 0)) {                                   \
            kpex::internal::check_fail(__func__, __FILE__, __LINE__, __VA_ARGS__); \
        }                                                                          \
    } while (0)

#define KPEX_CHECK_TENSOR_SHAPE(tensor, ...)                                                                    \
    KPEX_CHECK((tensor).sizes() == c10::IntArrayRef({__VA_ARGS__}), "invalid tensor shape: ", (tensor).sizes(), \
               ", expect: ", c10::IntArrayRef({__VA_ARGS__}))

#define KPEX_CHECK_TENSORWRAPPER_SHAPE(tensor, ...)                                 \
    KPEX_CHECK(c10::IntArrayRef((tensor).sizes) == c10::IntArrayRef({__VA_ARGS__}), \
               "invalid tensor wrapper shape: ", c10::IntArrayRef((tensor).sizes),  \
               ", expect: ", c10::IntArrayRef({__VA_ARGS__}))

#endif
