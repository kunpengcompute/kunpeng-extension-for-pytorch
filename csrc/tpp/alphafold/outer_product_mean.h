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

#ifndef KPEX_OUTER_PRODUCT_MEAN_H
#define KPEX_OUTER_PRODUCT_MEAN_H

#include <ATen/core/Tensor.h>
#include "kutacc.h"

namespace alphafold {

struct OuterProductMeanWeight {
    int64_t c_m;
    int64_t c_i;
    int64_t c_z;

    at::Tensor input_ln_w;
    at::Tensor input_ln_b;
    at::Tensor left_proj_w;
    at::Tensor left_proj_b;
    at::Tensor right_proj_w;
    at::Tensor right_proj_b;
    at::Tensor outer_w;
    at::Tensor outer_b;

    /**
         * @param input_ln_w shape[c_m]
         * @param input_ln_b shape[c_m]
         * @param left_proj_w shape [c_i, c_m]
         * @param left_proj_b shape [c_i]
         * @param right_proj_w shape [c_i. c_m]
         * @param right_proj_b shape [c_i]
         * @param output_w shape [c_z, c_i, c_i]
         * @param output_b shape [c_z]
         */
    OuterProductMeanWeight(at::Tensor &input_ln_w, at::Tensor &input_ln_b, at::Tensor &left_proj_w,
                           at::Tensor &left_proj_b, at::Tensor &right_proj_w, at::Tensor &right_proj_b,
                           at::Tensor &output_w, at::Tensor &output_b);
};

struct kutacc_af2_opm_weights_t_wrapper : kutacc_af2_opm_weights_t {
    kutacc_af2_opm_weights_t_wrapper(kutacc::TensorWrapper &left_proj_w, kutacc::TensorWrapper &left_proj_b,
                                     kutacc::TensorWrapper &right_proj_w, kutacc::TensorWrapper &right_proj_b,
                                     kutacc::TensorWrapper &outer_w, kutacc::TensorWrapper &outer_b, int64_t c_m,
                                     int64_t c_i, int64_t c_z)
    {
        this->left_proj_w = left_proj_w.get_tensor();
        this->left_proj_b = left_proj_b.get_tensor();
        this->right_proj_w = right_proj_w.get_tensor();
        this->right_proj_b = right_proj_b.get_tensor();
        this->outer_w = outer_w.get_tensor();
        this->outer_b = outer_b.get_tensor();
        this->c_m = c_m;
        this->c_i = c_i;
        this->c_z = c_z;
    }
};

struct kutacc_af2_opm_act_inputs_t_wrapper : kutacc_af2_opm_act_inputs_t {
    kutacc_af2_opm_act_inputs_t_wrapper(kutacc::TensorWrapper &input_act, kutacc::TensorWrapper &left_proj,
                                        kutacc::TensorWrapper &right_proj, kutacc::TensorWrapper &left_proj_,
                                        kutacc::TensorWrapper &right_proj_, int64_t n_seq, int64_t n_res)
    {
        this->input_act = input_act.get_tensor();
        this->left_proj = left_proj.get_tensor();
        this->right_proj = right_proj.get_tensor();
        this->left_proj_ = left_proj_.get_tensor();
        this->right_proj_ = right_proj_.get_tensor();
        this->n_seq = n_seq;
        this->n_res = n_res;
    }
};

struct kutacc_af2_opm_mask_inputs_t_wrapper : kutacc_af2_opm_mask_inputs_t {
    kutacc_af2_opm_mask_inputs_t_wrapper(kutacc::TensorWrapper &mask, kutacc::TensorWrapper &norm, int64_t n_res_gather,
                                         int64_t mask_bias)
    {
        this->mask = mask.get_tensor();
        this->norm = norm.get_tensor();
        this->n_res_gather = n_res_gather;
        this->mask_bias = mask_bias;
    }
};
/**
     * @param act shape [n_seq, n_res, c_m]
     * @param mask shape [n_seq, n_res_gather], n_res_gather is n_res before mpi chunk
     * @param weights
     * @param left_block_size left_block_size is used to set chunk size, it should be greater than 0 when its value is not null
     * @param right_block_size right_block_size is used to set chunk size, it should be greater than 0 when its value is not null
     * @return shape [n_res, n_res_gather, c_z]
     */
at::Tensor outer_product_mean(at::Tensor &act, at::Tensor &mask, const OuterProductMeanWeight &weights,
                              std::optional<int64_t> left_block_size, std::optional<int64_t> right_block_size,
                              bool no_mpi);

} // namespace alphafold

#endif
