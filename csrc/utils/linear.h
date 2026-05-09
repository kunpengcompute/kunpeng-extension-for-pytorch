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

#ifndef KPEX_LINEAR_H
#define KPEX_LINEAR_H

#include <vector>
#include <optional>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include "kutacc.h"

#include "check.h"
#include "TensorWrapper.h"

namespace alphafold {
/**
 * @brief calc act @ weight ^ T + bias + result
 * @param act shape [batch..., in_features]
 * @param weight shape [out_features..., in_features]
 * @param bias shape[out_features...]
 * @param result shape[batch..., out_features...]
 * @return shape[batch..., out_features]
 * bias.size = weight.size - 1
 * act.dim >= 2 weight.dim >= 2
 */
inline at::Tensor linear(const at::Tensor &act, const at::Tensor &weight, std::optional<at::Tensor> bias,
                         std::optional<at::Tensor> result_ = std::nullopt)
{
    KPEX_CHECK(act.dim() >= 2, act.dim());
    KPEX_CHECK(weight.dim() >= 2, weight.dim());
    if (bias.has_value()) {
        KPEX_CHECK(bias.value().dim() == weight.dim() - 1, bias.value().dim());
    }
    int64_t beta;
    at::Tensor result;
    if (!result_.has_value()) {
        auto result_sizes = act.sizes().vec();
        result_sizes.pop_back();
        result_sizes.insert(result_sizes.end(), weight.sizes().begin(), weight.sizes().end() - 1);
        result = at::empty(result_sizes, act.options());
        beta = 0;
    } else {
        result = result_.value();
        beta = 1;
    }
    if (bias.has_value()) {
        auto bias_ = bias.value();
        auto bias_data = (float *)bias_.data_ptr();
        auto act_tw = convert_to_tensor_wrapper(act);
        auto weight_tw = convert_to_tensor_wrapper(weight);
        auto result_tw = convert_to_tensor_wrapper(result);
        kutacc_af2_linear(act_tw.get_tensor(), weight_tw.get_tensor(), bias_data, result_tw.get_tensor(), beta);
        return result;
    } else {
        auto act_tw = convert_to_tensor_wrapper(act);
        auto weight_tw = convert_to_tensor_wrapper(weight);
        auto result_tw = convert_to_tensor_wrapper(result);
        kutacc_af2_linear(act_tw.get_tensor(), weight_tw.get_tensor(), nullptr, result_tw.get_tensor(), beta);
        return result;
    }
}
} // namespace alphafold

#endif
