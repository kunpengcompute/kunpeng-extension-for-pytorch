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

#ifndef KPEX_UTILS_MEMORY_H
#define KPEX_UTILS_MEMORY_H

#include <cstdint>
#include <memory>
#include <c10/core/Device.h>
#include <stddef.h>
#include <stdbool.h>

bool kpex_use_opm();

int kpex_posix_memalign(void **memptr, size_t alignment, size_t size);

void kpex_free(void *ptr);

namespace kpex {
inline c10::Device device() {
    return c10::Device(c10::kCPU, (int)kpex_use_opm());
}

template <typename T>
struct KpexMallocDeleter {
    void operator()(T *ptr) const
    {
        kpex_free(ptr);
    }
};

template <typename T>
inline std::unique_ptr<T[], KpexMallocDeleter<T> > alloc(int64_t size)
{
    void *ptr;
    kpex_posix_memalign(&ptr, 64, size * sizeof(T));
    return std::unique_ptr<T[], KpexMallocDeleter<T> >((T *)ptr);
}
}   // namespace kpex

#endif