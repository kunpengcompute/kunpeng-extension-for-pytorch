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

#include <ATen/ops/empty.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/record_function.h>
#include "kutacc.h"

#include "triangle_multiplication.h"
#include "utils/TensorWrapper.h"


namespace alphafold {

    at::Tensor triangle_multiplication(at::Tensor &act, at::Tensor &mask, const TriangleMultiplicationWeight &weights)
    {
        at::Tensor out = at::empty(act.sizes(), act.options());
        int64_t n_res = act.sizes()[0];
        int64_t n_res_gather = act.sizes()[1];
        KPEX_CHECK(act.dim() == 3, "act dim invalid, it should be 3 dims");
        if (world_size == 1) {
            KPEX_CHECK(n_res == n_res_gather, "act dim 0 size is not equal to dim 1 size. In single thread case, it will cause out-of-bounds mem access for gemm calc");
        }
        int64_t c_o = weights.c_o;
        int64_t c_i = weights.c_i;
        KPEX_CHECK_TENSOR_SHAPE(act, n_res, n_res_gather, c_o);
        KPEX_CHECK(act.dtype() == c10::kBFloat16, act.dtype());
        KPEX_CHECK(mask.dtype() == c10::kBFloat16, act.dtype());

        at::Tensor input_act = layernorm(act, weights.input_ln_w, weights.input_ln_b);
        at::Tensor center_act;
        at::Tensor left_proj_act = input_act.new_empty({c_i, n_res, n_res_gather});
        at::Tensor right_proj_act = input_act.new_empty({c_i, n_res, n_res_gather});
        at::Tensor gate = act.new_empty({n_res, n_res_gather, c_o});
        bool input_prepack = false;

        if (input_prepack) {
            input_act = linear_weight_prepack(input_act.view({n_res * n_res_gather, c_o}));
        }

        auto left_proj_act_tw = convert_to_tensor_wrapper(left_proj_act);
        auto right_proj_act_tw = convert_to_tensor_wrapper(right_proj_act);
        auto gate_tw = convert_to_tensor_wrapper(gate);
        auto input_act_tw = convert_to_tensor_wrapper(input_act);
        auto mask_tw = convert_to_tensor_wrapper(mask);
        auto left_proj_w_tw = convert_to_tensor_wrapper(weights.left_proj_w);
        auto left_proj_b_tw = convert_to_tensor_wrapper(weights.left_proj_b);
        auto left_gate_w_tw = convert_to_tensor_wrapper(weights.left_gate_w);
        auto left_gate_b_tw = convert_to_tensor_wrapper(weights.left_gate_b);
        auto right_proj_w_tw = convert_to_tensor_wrapper(weights.right_proj_w);
        auto right_proj_b_tw = convert_to_tensor_wrapper(weights.right_proj_b);
        auto right_gate_w_tw= convert_to_tensor_wrapper(weights.right_gate_w);
        auto right_gate_b_tw= convert_to_tensor_wrapper(weights.right_gate_b);

        at::Tensor gate_left = input_act.new_empty({c_i, n_res, n_res_gather});
        auto gate_left_tw = convert_to_tensor_wrapper(gate_left);

        kutacc_af2_tm_act_inputs_t_wrapper *left_acts_ptr = new kutacc_af2_tm_act_inputs_t_wrapper(left_proj_act_tw, input_act_tw, gate_left_tw, n_res, n_res_gather);
        kutacc_af2_tm_proj_weights_t_wrapper *left_weights_ptr = new kutacc_af2_tm_proj_weights_t_wrapper(left_proj_w_tw, left_proj_b_tw, left_gate_w_tw, left_gate_b_tw, c_o, c_i);
        if (unlikely(left_acts_ptr == nullptr || left_weights_ptr == nullptr)) {
            return out;
        }
        kutacc_af2_triangle_multiplication_calc_proj(left_acts_ptr, mask_tw.get_tensor(), left_weights_ptr, input_prepack);

        at::Tensor gate_right = input_act.new_empty({c_i, n_res, n_res_gather});
        auto gate_right_tw = convert_to_tensor_wrapper(gate_right);

        kutacc_af2_tm_act_inputs_t_wrapper *right_acts_ptr = new kutacc_af2_tm_act_inputs_t_wrapper(right_proj_act_tw, input_act_tw, gate_right_tw, n_res, n_res_gather);
        kutacc_af2_tm_proj_weights_t_wrapper *right_weights_ptr = new kutacc_af2_tm_proj_weights_t_wrapper(right_proj_w_tw, right_proj_b_tw, right_gate_w_tw, right_gate_b_tw, c_o, c_i);
        if (unlikely(right_acts_ptr == nullptr || right_weights_ptr == nullptr)) {
            return out;
        }
        kutacc_af2_triangle_multiplication_calc_proj(right_acts_ptr, mask_tw.get_tensor(), right_weights_ptr, input_prepack);

        if (n_res < n_res_gather) {
            left_proj_act = af2_transpose(left_proj_act, c_i, n_res_gather);
            right_proj_act = af2_transpose(right_proj_act, c_i, n_res_gather);
        }
        center_act = act.new_empty({left_proj_act.sizes()[0], n_res_gather, n_res_gather});

        auto center_act_tw = convert_to_tensor_wrapper(center_act);
        if (n_res < n_res_gather) {
            auto left_proj_act_new_tw = convert_to_tensor_wrapper(left_proj_act); // transpose后需要重新包装，规避内存重复释放问题
            auto right_proj_act_new_tw = convert_to_tensor_wrapper(right_proj_act);
            kutacc_af2_triangle_multiplication_equation(center_act_tw.get_tensor(), left_proj_act_new_tw.get_tensor(), right_proj_act_new_tw.get_tensor(), n_res_gather,
                weights.is_incoming);
            center_act = af2_transpose(center_act, c_i, n_res_gather);
        } else {
            kutacc_af2_triangle_multiplication_equation(center_act_tw.get_tensor(), left_proj_act_tw.get_tensor(), right_proj_act_tw.get_tensor(), n_res_gather,
                weights.is_incoming);
        }
        center_act = center_act.permute({1, 2, 0}).contiguous();
        center_act = layernorm(center_act, weights.center_ln_w, weights.center_ln_b);
        auto center_act_new_tw = convert_to_tensor_wrapper(center_act); // permute & layernorm之后重新包装center_act，规避内存重复释放问题

        auto out_tw = convert_to_tensor_wrapper(out);
        auto gating_w_tw = convert_to_tensor_wrapper(weights.gating_w);
        auto gating_b_tw = convert_to_tensor_wrapper(weights.gating_b);
        auto output_proj_w_tw = convert_to_tensor_wrapper(weights.output_proj_w);
        auto output_proj_b_tw = convert_to_tensor_wrapper(weights.output_proj_b);

        kutacc_af2_tm_linear_weights_t_wrapper *linear_weights_ptr = new kutacc_af2_tm_linear_weights_t_wrapper(gating_w_tw, gating_b_tw, output_proj_w_tw, output_proj_b_tw, c_o, c_i);
        if (unlikely(linear_weights_ptr == nullptr)) {
            return out;
        }
        kutacc_af2_triangle_multiplication_gate_and_out_linear(gate_tw.get_tensor(), out_tw.get_tensor(), left_acts_ptr, center_act_new_tw.get_tensor(), 
            linear_weights_ptr, input_prepack);
        kutacc_af2_triangle_multiplication_last(out_tw.get_tensor(), gate_tw.get_tensor(), n_res, n_res_gather, c_o);
        delete left_acts_ptr;
        delete left_weights_ptr;
        delete right_acts_ptr;
        delete right_weights_ptr;
        delete linear_weights_ptr;
        return out;
    }

    TriangleMultiplicationWeight::TriangleMultiplicationWeight(bool is_incoming, at::Tensor &input_ln_w, 
        at::Tensor &input_ln_b, at::Tensor &left_proj_w, at::Tensor &left_proj_b, at::Tensor &right_proj_w, 
        at::Tensor &right_proj_b, at::Tensor &left_gate_w, at::Tensor &left_gate_b, at::Tensor &right_gate_w, 
        at::Tensor &right_gate_b, at::Tensor &gating_w, at::Tensor &gating_b, at::Tensor &center_ln_w, 
        at::Tensor &center_ln_b, at::Tensor &output_proj_w, at::Tensor &output_proj_b)
    {
        KPEX_CHECK(input_ln_w.dim() == 1, "input_ln_w must have only one dim");
        KPEX_CHECK(left_proj_w.dim() == 2, "left_proj_w must have two dims");
        int64_t c_o = input_ln_w.sizes()[0];
        int64_t c_i = left_proj_w.sizes()[0];
        KPEX_CHECK(c_o > 0 && c_i > 0 && c_o <= INT64_MAX, "invalid input_ln_w shape & invalid left_proj_w shape");
        KPEX_CHECK(c_i == c_o && c_i <= INT64_MAX / c_o, "left_proj_w dim 0 size is not equal to input_ln_w dim 0 size, it will cause calc error");
        KPEX_CHECK_TENSOR_SHAPE(input_ln_b, c_o);
        KPEX_CHECK_TENSOR_SHAPE(left_proj_w, c_i, c_o);
        KPEX_CHECK_TENSOR_SHAPE(left_proj_b, c_i);
        KPEX_CHECK_TENSOR_SHAPE(right_proj_w, c_i, c_o);
        KPEX_CHECK_TENSOR_SHAPE(right_proj_b, c_i);
        KPEX_CHECK_TENSOR_SHAPE(left_gate_w, c_i, c_o);
        KPEX_CHECK_TENSOR_SHAPE(left_gate_b, c_i);
        KPEX_CHECK_TENSOR_SHAPE(right_gate_w, c_i, c_o);
        KPEX_CHECK_TENSOR_SHAPE(right_gate_b, c_i);
        KPEX_CHECK_TENSOR_SHAPE(gating_w, c_i, c_o);
        KPEX_CHECK_TENSOR_SHAPE(gating_b, c_i);
        KPEX_CHECK_TENSOR_SHAPE(center_ln_w, c_i);
        KPEX_CHECK_TENSOR_SHAPE(center_ln_b, c_i);
        KPEX_CHECK_TENSOR_SHAPE(output_proj_w, c_o, c_i);
        KPEX_CHECK_TENSOR_SHAPE(output_proj_b, c_o);

        auto float_opt = left_proj_w.options().device(kpex::device()).dtype(c10::kFloat);
        auto bf16_opt = left_proj_w.options().device(kpex::device()).dtype(c10::kBFloat16);
        left_proj_w = left_proj_w.to(bf16_opt).contiguous();
        right_proj_w = right_proj_w.to(bf16_opt).contiguous();
        left_gate_w = left_gate_w.to(bf16_opt).contiguous();
        right_gate_w = right_gate_w.to(bf16_opt).contiguous();
        gating_w = gating_w.to(bf16_opt).contiguous();
        output_proj_w = output_proj_w.to(bf16_opt).contiguous();

        this->c_i = c_i;
        this->c_o = c_o;
        this->is_incoming = is_incoming;
        this->left_proj_w = linear_weight_prepack(left_proj_w);
        this->right_proj_w = linear_weight_prepack(right_proj_w);
        this->left_gate_w = linear_weight_prepack(left_gate_w);
        this->right_gate_w = linear_weight_prepack(right_gate_w);
        this->gating_w = linear_weight_prepack(gating_w);
        this->output_proj_w = linear_weight_prepack(output_proj_w);

        this->input_ln_w = input_ln_w.to(float_opt).contiguous();
        this->input_ln_b = input_ln_b.to(float_opt).contiguous();
        this->left_proj_b = left_proj_b.to(float_opt).contiguous();
        this->right_proj_b = right_proj_b.to(float_opt).contiguous();
        this->left_gate_b = left_gate_b.to(float_opt).contiguous();
        this->right_gate_b = right_gate_b.to(float_opt).contiguous();
        this->gating_b = gating_b.to(float_opt).contiguous();
        this->center_ln_w = center_ln_w.to(float_opt).contiguous();
        this->center_ln_b = center_ln_b.to(float_opt).contiguous();
        this->output_proj_b = output_proj_b.to(float_opt).contiguous();
    }
}