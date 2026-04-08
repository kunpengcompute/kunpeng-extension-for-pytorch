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

#ifndef KPEX_TPP_ALPHAFOLD_INVARIANT_POINT_H
#define KPEX_TPP_ALPHAFOLD_INVARIANT_POINT_H

#include <ATen/core/Tensor.h>
#include "kutacc.h"
#include "utils/check.h"

namespace alphafold {
struct InvariantPointAttentionWeight {
    int64_t c_s;
    int64_t c_z;
    int64_t c_hidden;
    int64_t no_heads;
    int64_t no_qk_points;
    int64_t no_v_points;

    at::Tensor linear_q_w;
    at::Tensor linear_q_b;
    at::Tensor linear_k_w;
    at::Tensor linear_k_b;
    at::Tensor linear_v_w;
    at::Tensor linear_v_b;
    at::Tensor linear_q_points_w;
    at::Tensor linear_q_points_b;
    at::Tensor linear_k_points_w;
    at::Tensor linear_k_points_b;
    at::Tensor linear_v_points_w;
    at::Tensor linear_v_points_b;
    at::Tensor linear_b_w;
    at::Tensor linear_b_b;
    at::Tensor head_weights;
    at::Tensor linear_out_w;
    at::Tensor linear_out_b;

    InvariantPointAttentionWeight(int64_t c_s, int64_t c_z, int64_t c_hidden, int64_t no_heads, int64_t no_qk_points,
        int64_t no_v_points, at::Tensor &linear_q_w, at::Tensor &linear_q_b, at::Tensor &linear_kv_w, at::Tensor linear_kv_b,
        at::Tensor &linear_q_points_w, at::Tensor &linear_q_points_b, at::Tensor &linear_kv_points_w, at::Tensor &linear_kv_points_b,
        at::Tensor &linear_b_w, at::Tensor &linear_b_b, at::Tensor &head_weights, at::Tensor &linear_out_w, at::Tensor &linear_out_b);
};

struct kutacc_af2_ipa_weights_t_wrapper : kutacc_af2_ipa_weights_t {
    kutacc_af2_ipa_weights_t_wrapper(kutacc::TensorWrapper &head_weights, kutacc::TensorWrapper &weights_head_weights, 
        kutacc::TensorWrapper &linear_b_w, kutacc::TensorWrapper &linear_b_b, int64_t c_z, int64_t c_hidden,
        int64_t no_heads, int64_t no_qk_points, int64_t no_v_points)
    {
        this->head_weights = head_weights.get_tensor();
        this->weights_head_weights = weights_head_weights.get_tensor();
        this->linear_b_w = linear_b_w.get_tensor();
        this->linear_b_b = linear_b_b.get_tensor();
        this->c_z = c_z;
        this->c_hidden = c_hidden;
        this->no_heads = no_heads;
        this->no_qk_points = no_qk_points;
        this->no_v_points = no_v_points;
    }
};

struct kutacc_af2_ipa_s_inputs_t_wrapper : kutacc_af2_ipa_s_inputs_t {
    kutacc_af2_ipa_s_inputs_t_wrapper(kutacc::TensorWrapper &a, kutacc::TensorWrapper &b, kutacc::TensorWrapper &q, kutacc::TensorWrapper &k,
        kutacc::TensorWrapper &v, kutacc::TensorWrapper &q_pts, kutacc::TensorWrapper &k_pts, kutacc::TensorWrapper &v_pts, int64_t n_res) 
    {
        this->a = a.get_tensor();
        this->b = b.get_tensor();
        this->q = q.get_tensor();
        this->k = k.get_tensor();
        this->v = v.get_tensor();
        this->q_pts = q_pts.get_tensor();
        this->k_pts = k_pts.get_tensor();
        this->v_pts = v_pts.get_tensor();
        this->n_res = n_res;
    }
};

struct kutacc_af2_ipa_o_inputs_t_wrapper : kutacc_af2_ipa_o_inputs_t {
    kutacc_af2_ipa_o_inputs_t_wrapper(kutacc::TensorWrapper &o, kutacc::TensorWrapper &o_pt, kutacc::TensorWrapper &o_pt_norm, kutacc::TensorWrapper &o_pair)
    {
        this->o = o.get_tensor();
        this->o_pt = o_pt.get_tensor();
        this->o_pt_norm = o_pt_norm.get_tensor();
        this->o_pair = o_pair.get_tensor();
    }
};
    /**
     * @param s [n_res, c_s]
     * @param z [n_res, n_res, c_z]
     * @param rigid_trans [n_res, 3]
     * @param rigid_rot_mats [n_res, 3, 3]
     * @param mask [n_res]
     */
    at::Tensor invariant_point_attention(at::Tensor &s, at::Tensor &z, at::Tensor &rigid_trans, at::Tensor &rigid_rot_mats, 
        at::Tensor &mask, const InvariantPointAttentionWeight &weights);
}

#endif