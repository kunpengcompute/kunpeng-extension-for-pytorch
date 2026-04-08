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
#include "kutacc.h"

#include "outer_product_mean.h"
#include "utils/all_gather.h"
#include "utils/check.h"
#include "utils/layernorm.h"
#include "utils/TensorWrapper.h"
#include "utils/memory.h"

namespace alphafold {

namespace {
void default_block_size(int64_t n_seq, int64_t n_res_gather, int64_t &left_block_size, int64_t &right_block_size)
{
    if (n_seq == 512) {
        if (n_res_gather < 200) {
            left_block_size = 4;
        } else if (n_res_gather < 400) {
            left_block_size = 8;
        } else {
            left_block_size = 16;
        }
        if (n_res_gather < 200) {
            right_block_size = 16;
        } else if (n_res_gather < 300) {
            right_block_size = 32;
        } else if (n_res_gather < 700) {
            right_block_size = 80;
        } else {
            right_block_size = 112;
        }
    } else {
        left_block_size = std::clamp(n_res_gather / 24, (int64_t)8, (int64_t)80);
        right_block_size = std::clamp(n_res_gather / 8, (int64_t)16, (int64_t)192);
    }
}
}

OuterProductMeanWeight::OuterProductMeanWeight(at::Tensor &input_ln_w, at::Tensor &input_ln_b, at::Tensor &left_proj_w,
                                               at::Tensor &left_proj_b, at::Tensor &right_proj_w,
                                               at::Tensor &right_proj_b, at::Tensor &output_w, at::Tensor &output_b)
{
    KPEX_CHECK(input_ln_w.dim() == 1, "input_ln_w must have only 1 dim");
    KPEX_CHECK(left_proj_w.dim() == 2, "left_proj_w must have 2 dims");
    KPEX_CHECK(output_w.dim() == 3, "output_w must have 3 dims");
    int64_t c_m = input_ln_w.sizes()[0];
    int64_t c_i = left_proj_w.sizes()[0];
    int64_t c_z = output_w.sizes()[0];
    // c_i ^ 2 * left_block_size_max(80) * right_block_size_max(192) < INT64_MAX
    // c_z * left_block_size_max(80) * right_block_size_max(192) < INT64_MAX
    KPEX_CHECK(c_m > 0 && c_i > 0 && c_z > 0 && c_i <= std::sqrt(INT64_MAX / (80 * 192)) && c_z <= INT64_MAX / (80 * 192) && c_m <= INT64_MAX && c_m <= INT64_MAX / c_i,
        "invalid input_ln_w shape & invalid left_proj_w shape & invalid output_w shape");
    KPEX_CHECK_TENSOR_SHAPE(input_ln_b, c_m);
    KPEX_CHECK_TENSOR_SHAPE(left_proj_w, c_i, c_m);
    KPEX_CHECK_TENSOR_SHAPE(left_proj_b, c_i);
    KPEX_CHECK_TENSOR_SHAPE(right_proj_w, c_i, c_m);
    KPEX_CHECK_TENSOR_SHAPE(right_proj_b, c_i);
    KPEX_CHECK_TENSOR_SHAPE(output_w, c_z, c_i, c_i);
    KPEX_CHECK_TENSOR_SHAPE(output_b, c_z);

    auto float_opt = left_proj_w.options().device(kpex::device()).dtype(c10::kFloat);
    auto bf16_opt = left_proj_w.options().device(kpex::device()).dtype(c10::kBFloat16);
    this->left_proj_w = left_proj_w.to(bf16_opt).contiguous();
    this->right_proj_w = right_proj_w.to(bf16_opt).contiguous();
    this->outer_w = output_w.to(bf16_opt).contiguous().view({c_z, c_i * c_i});
    this->c_m = c_m;
    this->c_i = c_i;
    this->c_z = c_z;
    auto left_proj_w_res = linear_weight_prepack(this->left_proj_w);
    auto right_proj_w_res = linear_weight_prepack(this->right_proj_w);
    auto outer_w_res = linear_weight_prepack(this->outer_w);
    this->left_proj_w = left_proj_w_res;
    this->right_proj_w = right_proj_w_res;
    this->outer_w = outer_w_res;

    this->input_ln_w = input_ln_w.to(float_opt).contiguous();
    this->input_ln_b = input_ln_b.to(float_opt).contiguous();
    this->left_proj_b = left_proj_b.to(float_opt).contiguous();
    this->right_proj_b = right_proj_b.to(float_opt).contiguous();
    this->outer_b = output_b.to(float_opt).contiguous();
}

at::Tensor outer_product_mean(at::Tensor &act, at::Tensor &mask, const OuterProductMeanWeight &weights,
                              std::optional<int64_t> left_block_size_, std::optional<int64_t> right_block_size_,
                              bool no_mpi)
{
    at::Tensor out = act.new_empty({act.sizes()[1], mask.sizes()[1], weights.c_z});
    int64_t n_seq = act.sizes()[0];
    int64_t n_res = act.sizes()[1];
    int64_t n_res_gather = mask.sizes()[1];
    if (world_size == 1) {
        // in single process case: n_res should be equal to n_res_gather, otherwise will cause out-of-bound access for gemm calculation
        KPEX_CHECK(n_res == n_res_gather, "act dim 1 isn't equal to mask dim 1");
    }
    int64_t c_m = weights.c_m;
    int64_t c_i = weights.c_i;
    int64_t c_z = weights.c_z;
    int64_t left_block_size;
    int64_t right_block_size;
    default_block_size(n_seq, n_res_gather, left_block_size, right_block_size);
    left_block_size = left_block_size_.value_or(left_block_size);
    right_block_size = right_block_size_.value_or(right_block_size);

    KPEX_CHECK(left_block_size > 0 && right_block_size > 0, "left_block_size or right_block_size values are less than or equal to zero\n");
    KPEX_CHECK(act.dtype() == c10::kBFloat16, act.dtype());
    KPEX_CHECK(mask.dtype() == c10::kBFloat16, mask.dtype());
    KPEX_CHECK_TENSOR_SHAPE(act, n_seq, n_res, c_m);
    KPEX_CHECK_TENSOR_SHAPE(mask, n_seq, n_res_gather);
    act = act.contiguous();
    mask = mask.transpose(0, 1).contiguous();

    at::Tensor left_proj = act.new_empty({c_i, n_res, n_seq});
    at::Tensor right_proj = act.new_empty({c_i, n_res, n_seq});
    at::Tensor left_proj_ = act.new_empty({n_res, c_i, n_seq});
    at::Tensor right_proj_ = act.new_empty({n_res, c_i, n_seq});
    at::Tensor norm = mask.new_empty({n_res, n_res_gather});
    int64_t mask_bias = 0;

    if (n_res_gather > n_res) {
        mask_bias = rank * ((n_res_gather + world_size - 1) / world_size) * mask.strides()[0];
    }

    at::Tensor input_act = layernorm(act.transpose(0, 1), weights.input_ln_w, weights.input_ln_b);

    auto input_act_tw = convert_to_tensor_wrapper(input_act);
    auto mask_tw = convert_to_tensor_wrapper(mask);
    auto left_proj_w_tw = convert_to_tensor_wrapper(weights.left_proj_w);
    auto left_proj_b_tw = convert_to_tensor_wrapper(weights.left_proj_b);
    auto right_proj_w_tw = convert_to_tensor_wrapper(weights.right_proj_w);
    auto right_proj_b_tw = convert_to_tensor_wrapper(weights.right_proj_b);
    auto left_proj_tw = convert_to_tensor_wrapper(left_proj);
    auto right_proj_tw = convert_to_tensor_wrapper(right_proj);
    auto left_proj_tw_ = convert_to_tensor_wrapper(left_proj_);
    auto right_proj_tw_ = convert_to_tensor_wrapper(right_proj_);
    auto norm_tw = convert_to_tensor_wrapper(norm);
    auto output_w_tw = convert_to_tensor_wrapper(weights.outer_w);
    auto output_b_tw = convert_to_tensor_wrapper(weights.outer_b);
    auto out_tw = convert_to_tensor_wrapper(out);

    kutacc_af2_opm_weights_t_wrapper *opm_weights_ptr = new kutacc_af2_opm_weights_t_wrapper(left_proj_w_tw, left_proj_b_tw, right_proj_w_tw, right_proj_b_tw,
        output_w_tw, output_b_tw, c_m, c_i, c_z);
    kutacc_af2_opm_act_inputs_t_wrapper *opm_inputs_ptr = new kutacc_af2_opm_act_inputs_t_wrapper(input_act_tw, left_proj_tw, right_proj_tw, left_proj_tw_,
        right_proj_tw_, n_seq, n_res);
    kutacc_af2_opm_mask_inputs_t_wrapper *opm_mask_ptr = new kutacc_af2_opm_mask_inputs_t_wrapper(mask_tw, norm_tw, n_res_gather, mask_bias);

    if (unlikely(opm_weights_ptr == nullptr || opm_inputs_ptr == nullptr || opm_mask_ptr == nullptr)) {
        return out;
    }
    kutacc_af2_outer_product_mean_calc_left_and_right_mul(opm_inputs_ptr, opm_mask_ptr, opm_weights_ptr);

    if (n_res_gather > n_res) {
        if (!no_mpi) {
            right_proj_ = af2_all_gather(right_proj_, n_res_gather, c_i);
        } else {
            right_proj_ = at::empty({n_res_gather, c_i, n_seq}, right_proj_.options());
        }

        auto all_gather_right_proj_tw_ = convert_to_tensor_wrapper(right_proj_);

        kutacc_af2_opm_act_inputs_t_wrapper *opm_all_gather_inputs_ptr = new kutacc_af2_opm_act_inputs_t_wrapper(input_act_tw, left_proj_tw, right_proj_tw, left_proj_tw_,
        all_gather_right_proj_tw_, n_seq, n_res);
        if (unlikely(opm_all_gather_inputs_ptr == nullptr)) {
            return out;
        }
        kutacc_af2_outer_product_mean_chunk(opm_all_gather_inputs_ptr, opm_mask_ptr, opm_weights_ptr, out_tw.get_tensor(), left_block_size, right_block_size);
        delete opm_all_gather_inputs_ptr;
    } else {
        kutacc_af2_outer_product_mean_chunk(opm_inputs_ptr, opm_mask_ptr, opm_weights_ptr, out_tw.get_tensor(), left_block_size, right_block_size);
    }
    delete opm_inputs_ptr;
    delete opm_weights_ptr;
    delete opm_mask_ptr;
    return out;
}

}