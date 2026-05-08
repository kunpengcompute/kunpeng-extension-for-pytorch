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

#ifndef KPEX_UTILS_LAYERNORM_H
#define KPEX_UTILS_LAYERNORM_H

#include <vector>
#include <ATen/core/Tensor.h>
#include <ATen/native/cpu/utils.h>

#include "kutacc.h"
#include "parallel.h"

/**
 * @brief layernorm return a contiguous value
 * @param act shape [m, n, len]
 * @param weight shape [len]
 * @param bias shape [len]
 * @return shape [m, n, len]
 */
inline at::Tensor layernorm(const at::Tensor &act, const at::Tensor &weight_, const at::Tensor &bias_)
{
    KPEX_CHECK(act.dim() == 3, act.sizes());
    int64_t m = act.sizes()[0];
    int64_t n = act.sizes()[1];
    int64_t len = act.sizes()[2];
    KPEX_CHECK(act.strides()[2] == 1, act.strides());
    KPEX_CHECK(act.dtype() == c10::kBFloat16, act.dtype());
    KPEX_CHECK_TENSOR_SHAPE(weight_, len);
    KPEX_CHECK_TENSOR_SHAPE(bias_, len);
    auto weight = weight_.to(c10::kFloat);
    auto bias = bias_.to(c10::kFloat);
    auto out = act.new_empty(act.sizes());
    kpex::parallel_for(0, m * n, 1, [&](int64_t start, int64_t end) {
        int64_t mi, ni;
        at::native::data_index_init(start, mi, m, ni, n);
        for ([[maybe_unused]] int64_t _ : c10::irange(start, end)) {
            kutacc_af2_layernorm((__bf16 *)act.data_ptr() + mi * act.strides()[0] + ni * act.strides()[1],
                                 (float *)weight.data_ptr(), (float *)bias.data_ptr(), len, 1e-5,
                                 (__bf16 *)out.data_ptr() + mi * out.strides()[0] + ni * out.strides()[1]);
            at::native::data_index_step(mi, m, ni, n);
        }
    });
    return out;
}

#endif
