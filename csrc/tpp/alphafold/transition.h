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

#ifndef KPEX_TPP_ALPHAFOLD_TRANSITION_H
#define KPEX_TPP_ALPHAFOLD_TRANSITION_H

#include <ATen/core/Tensor.h>
#include "kutacc.h"
#include "utils/check.h"

namespace alphafold {
struct TransitionWeight {
    int64_t c_o;
    int64_t c_i;

    at::Tensor input_ln_w;
    at::Tensor input_ln_b;
    at::Tensor linear1_w;
    at::Tensor linear1_b;
    at::Tensor linear2_w;
    at::Tensor linear2_b;

    /**
    * @param input_ln_w shape [c_o]
    * @param input_ln_b shape [c_o]
    * @param linear1_w shape [c_i, c_o]
    * @param linear1_b shape [c_i]
    * @param linear2_w shape [c_o, c_i]
    * @param linear2_b shape [c_o]
    */
    TransitionWeight(at::Tensor &input_ln_w,at::Tensor &input_ln_b, at::Tensor &linear1_w, at::Tensor &linear1_b,
        at::Tensor &linear2_w, at::Tensor &linear2_b);
};

struct kutacc_af2_trans_weights_t_wrapper : kutacc_af2_trans_weights_t {
    kutacc_af2_trans_weights_t_wrapper(kutacc::TensorWrapper &linear1_w, kutacc::TensorWrapper &linear1_b, kutacc::TensorWrapper &linear2_w, kutacc::TensorWrapper &linear2_b, int64_t c_o, int64_t c_i)
    {
        this->linear1_w = linear1_w.get_tensor();
        this->linear1_b = linear1_b.get_tensor();
        this->linear2_w = linear2_w.get_tensor();
        this->linear2_b = linear2_b.get_tensor();
        this->c_o = c_o;
        this->c_i = c_i;
    }
};

struct kutacc_af2_trans_act_inputs_t_wrapper : kutacc_af2_trans_act_inputs_t {
    kutacc_af2_trans_act_inputs_t_wrapper(kutacc::TensorWrapper &input_act, kutacc::TensorWrapper &intermediate_act, int64_t batch, int64_t n_res)
    {
        this->input_act = input_act.get_tensor();
        this->intermediate_act= intermediate_act.get_tensor();
        this->batch = batch;
        this->n_res = n_res;
    }
};
/**
 * @param act shape [batch, n_res, c_o]
 * @param weights
 * @return shape [batch, n_res, c_o]
 */
at::Tensor transition(at::Tensor &act, const TransitionWeight &weights);
}

#endif
