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

#ifndef KPEX_ALL_GATHER_H
#define KPEX_ALL_GATHER_H

#include <vector>
#include "kutacc.h"
#include <ATen/core/Tensor.h>
#include <ATen/native/cpu/utils.h>
#include "TensorWrapper.h"
#include "check.h"

inline at::Tensor af2_all_gather(at::Tensor &data, int64_t m, int64_t n)
{
    int64_t len = data.sizes()[2];

    KPEX_CHECK(data.strides()[2] == 1, data.strides()[2]);

    at::Tensor out = at::empty({m, n, len}, data.options());

    auto data_tw = convert_to_tensor_wrapper_comm(data);
    auto out_tw = convert_to_tensor_wrapper_comm(out);

    kutacc_af2_all_gather(data_tw.get_tensor(), out_tw.get_tensor());
    return out;
}

#endif
