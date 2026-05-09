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

#ifndef KPEX_TRANSPOSE_H
#define KPEX_TRANSPOSE_H

#include <vector>
#include <ATen/core/Tensor.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/ops/empty.h>
#include "kutacc.h"

#include "TensorWrapper.h"
#include "check.h"

inline at::Tensor af2_transpose(at::Tensor &data, int64_t m, int64_t n)
{
    int64_t block_m = (m + world_size - 1) / world_size;
    int64_t block_n = (n + world_size - 1) / world_size;
    int64_t len = data.sizes()[2];

    KPEX_CHECK(data.strides()[2] == 1, data.strides()[2]);

    at::Tensor out;
    if (data.sizes()[0] < m) {
        out = at::empty({m, std::min(block_n, n - rank * block_n), len}, data.options());
    } else {
        out = at::empty({std::min(block_m, m - rank * block_m), n, len}, data.options());
    }

    auto data_tw = convert_to_tensor_wrapper_comm(data);
    auto out_tw = convert_to_tensor_wrapper_comm(out);

    kutacc_af2_transpose(data_tw.get_tensor(), out_tw.get_tensor());
    return out;
}

#endif
