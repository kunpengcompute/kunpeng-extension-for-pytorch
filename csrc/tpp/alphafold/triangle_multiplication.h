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

#ifndef KPEX_TPP_ALPHAFOLD_TRIANGLE_MULTIPLICATION_H
#define KPEX_TPP_ALPHAFOLD_TRIANGLE_MULTIPLICATION_H

#include <ATen/core/Tensor.h>
#include "kutacc.h"

#include "utils/check.h"
#include "utils/layernorm.h"
#include "utils/memory.h"
#include "utils/transpose.h"

namespace alphafold {

    struct TriangleMultiplicationWeight {
        int64_t c_o;
        int64_t c_i;

        bool is_incoming;
        at::Tensor input_ln_w;
        at::Tensor input_ln_b;
        at::Tensor left_proj_w;
        at::Tensor left_proj_b;
        at::Tensor right_proj_w;
        at::Tensor right_proj_b;
        at::Tensor left_gate_w;
        at::Tensor left_gate_b;
        at::Tensor right_gate_w;
        at::Tensor right_gate_b;
        at::Tensor gating_w;
        at::Tensor gating_b;
        at::Tensor center_ln_w;
        at::Tensor center_ln_b;
        at::Tensor output_proj_w;
        at::Tensor output_proj_b;

        /**
         * @param is_incoming outgoing equation 'ikc,jkc->ijc'. incoming equation 'kjc,kic->ijc'
         * @param input_ln_w shape [c_o]
         * @param input_ln_b shape [c_o]
         * @param left_proj_w shape [c_i, c_o]
         * @param left_proj_b shape [c_i]
         * @param right_proj_w shape [c_i, c_o]
         * @param right_proj_b shape [c_i]
         * @param left_gate_w shape [c_i, c_o]
         * @param left_gate_b shape [c_i]
         * @param right_gate_w shape [c_i, c_o]
         * @param right_gate_b shape [c_i]
         * @param gating_w shape [c_i, c_o]
         * @param gating_b shape [c_i]
         * @param center_ln_w shape [c_i]
         * @param center_ln_b shape [c_i]
         * @param output_proj_w shape [c_o, c_i]
         * @param output_proj_b shape [c_o]
         */
        TriangleMultiplicationWeight(bool is_incoming, at::Tensor &input_ln_w, at::Tensor &input_ln_b, 
            at::Tensor &left_proj_w, at::Tensor &left_proj_b, at::Tensor &right_proj_w, at::Tensor &right_proj_b,
            at::Tensor &left_gate_w, at::Tensor &left_gate_b, at::Tensor &right_gate_w, at::Tensor &right_gate_b, 
            at::Tensor &gating_w, at::Tensor &gating_b, at::Tensor &center_ln_w, at::Tensor &center_ln_b, 
            at::Tensor &output_proj_w, at::Tensor &output_proj_b);

    };

    struct kutacc_af2_tm_proj_weights_t_wrapper : kutacc_af2_tm_proj_weights_t {
        kutacc_af2_tm_proj_weights_t_wrapper(kutacc::TensorWrapper &proj_w, kutacc::TensorWrapper &proj_b, kutacc::TensorWrapper &gate_w, kutacc::TensorWrapper &gate_b, int64_t c_o, int64_t c_i)
        {
            this->proj_w = proj_w.get_tensor();
            this->proj_b = proj_b.get_tensor();
            this->gate_w = gate_w.get_tensor();
            this->gate_b = gate_b.get_tensor();
            this->c_o = c_o;
            this->c_i = c_i;
        }
    };

    struct kutacc_af2_tm_linear_weights_t_wrapper : kutacc_af2_tm_linear_weights_t {
        kutacc_af2_tm_linear_weights_t_wrapper(kutacc::TensorWrapper &gating_w, kutacc::TensorWrapper &gating_b,kutacc::TensorWrapper &output_proj_w, kutacc::TensorWrapper &output_proj_b, int64_t c_o, int64_t c_i)
        {
            this->gating_w = gating_w.get_tensor();
            this->gating_b = gating_b.get_tensor();
            this->output_proj_w = output_proj_w.get_tensor();
            this->output_proj_b = output_proj_b.get_tensor();
            this->c_o = c_o;
            this->c_i = c_i;
        }

    };

    struct kutacc_af2_tm_act_inputs_t_wrapper : kutacc_af2_tm_act_inputs_t {
        kutacc_af2_tm_act_inputs_t_wrapper(kutacc::TensorWrapper &proj_act, kutacc::TensorWrapper &input_act, kutacc::TensorWrapper &proj_act_gate, 
            int64_t n_res, int64_t n_res_gather)
        {
            this->proj_act = proj_act.get_tensor();
            this->input_act = input_act.get_tensor();
            this->proj_act_gate = proj_act_gate.get_tensor();
            this->n_res = n_res;
            this->n_res_gather = n_res_gather;
        }
    };

    /**
     * @param act shape [n_res, n_res_gather, c_z]
     * @param mask shape [n_res, n_res_gather]
     * @param weights
     */
    at::Tensor triangle_multiplication(at::Tensor &act, at::Tensor &mask, const TriangleMultiplicationWeight &weights);

}

#endif