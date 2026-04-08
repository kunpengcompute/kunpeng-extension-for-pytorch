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

#include <ATen/record_function.h>
#include <ATen/ops/empty.h>

#include "transition.h"
#include "utils/bf16.h"
#include "utils/TensorWrapper.h"
#include "utils/layernorm.h"
#include "utils/memory.h"

namespace alphafold {
at::Tensor transition(at::Tensor &act, const TransitionWeight &weights)
{
    at::Tensor out = act.new_empty(act.sizes());
    
    int64_t batch = act.sizes()[0];
    int64_t n_res = act.sizes()[1];
    KPEX_CHECK(batch > 0 && n_res > 0 && batch <= INT64_MAX && n_res <= INT64_MAX / batch, "act sizes[0] * act_sizes[1] will cause overflow");
    int64_t c_o = weights.c_o;
    int64_t c_i = weights.c_i;
    KPEX_CHECK(act.dtype() == c10::kBFloat16, act.dtype());
    KPEX_CHECK(act.dim() == 3 && act.sizes()[0] == batch && act.sizes()[1] == n_res && act.sizes()[2] == c_o, act.sizes());
    act = act.contiguous();
    at::Tensor input_act = layernorm(act, weights.input_ln_w, weights.input_ln_b);
    at::Tensor intermediate_act = act.new_empty({batch * n_res, c_i});

    auto input_act_tw = convert_to_tensor_wrapper(input_act);
    auto linear1_w_tw = convert_to_tensor_wrapper(weights.linear1_w);
    auto linear1_b_tw = convert_to_tensor_wrapper(weights.linear1_b);
    auto linear2_w_tw = convert_to_tensor_wrapper(weights.linear2_w);
    auto linear2_b_tw = convert_to_tensor_wrapper(weights.linear2_b);
    auto intermediate_act_tw = convert_to_tensor_wrapper(intermediate_act);
    auto out_tw = convert_to_tensor_wrapper(out);

    kutacc_af2_trans_weights_t_wrapper *trans_weights_ptr = new kutacc_af2_trans_weights_t_wrapper(linear1_w_tw, linear1_b_tw, linear2_w_tw, linear2_b_tw, c_o, c_i);
    kutacc_af2_trans_act_inputs_t_wrapper *trans_inputs_ptr = new kutacc_af2_trans_act_inputs_t_wrapper(input_act_tw, intermediate_act_tw, batch, n_res);

    if (unlikely(trans_weights_ptr == nullptr || trans_inputs_ptr == nullptr)) {
        return out;
    }

    kutacc_af2_transition(trans_inputs_ptr, trans_weights_ptr, out_tw.get_tensor());
    delete trans_weights_ptr;
    delete trans_inputs_ptr;
    return out;
}

TransitionWeight::TransitionWeight(at::Tensor &input_ln_w, at::Tensor &input_ln_b, at::Tensor &linear1_w,
    at::Tensor &linear1_b, at::Tensor &linear2_w, at::Tensor &linear2_b)
{
    KPEX_CHECK(input_ln_w.dim() == 1, "input_ln_w must have only 1 dim");
    KPEX_CHECK(linear1_w.dim() == 2, "linear1_w must have 2 dims");
    int64_t c_o = input_ln_w.sizes()[0];
    int64_t c_i = linear1_w.sizes()[0];
    KPEX_CHECK(c_o > 0 && c_i > 0 && c_i <= INT64_MAX && c_i <= INT64_MAX / c_o, "invalid input_ln_w shape and invalid linear_1_w shape");
    KPEX_CHECK_TENSOR_SHAPE(input_ln_w, c_o);
    KPEX_CHECK_TENSOR_SHAPE(input_ln_b, c_o);
    KPEX_CHECK_TENSOR_SHAPE(linear1_w, c_i, c_o);
    KPEX_CHECK_TENSOR_SHAPE(linear1_b, c_i);
    KPEX_CHECK_TENSOR_SHAPE(linear2_w, c_o, c_i);
    KPEX_CHECK_TENSOR_SHAPE(linear2_b, c_o);

    auto float_opt = linear1_w.options().device(kpex::device()).dtype(c10::kFloat);
    auto bf16_opt = linear1_w.options().device(kpex::device()).dtype(c10::kBFloat16);

    linear1_w = linear1_w.to(bf16_opt).contiguous();
    linear2_w = linear2_w.to(bf16_opt).contiguous();

    this->c_i = c_i;
    this->c_o = c_o;
    this->linear1_w = linear_weight_prepack(linear1_w);
    this->linear2_w = linear_weight_prepack(linear2_w);
    
    this->input_ln_w = input_ln_w.to(float_opt).contiguous();
    this->input_ln_b = input_ln_b.to(float_opt).contiguous();
    this->linear1_b = linear1_b.to(float_opt).contiguous();
    this->linear2_b = linear2_b.to(float_opt).contiguous();
}
}