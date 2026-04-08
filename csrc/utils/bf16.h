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

#ifndef KPEX_UTILS_BF16_H
#define KPEX_UTILS_BF16_H

#include <arm_neon.h>
#include <cstdint>

namespace kpex {
static inline __bf16 to_bf16(float x)
{
    return vcvth_bf16_f32(x);
}

static inline float to_float(__bf16 x)
{
    return vcvtah_f32_bf16(x);
}
}   // namespace kpex

#endif