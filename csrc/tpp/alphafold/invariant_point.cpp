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

#include <ATen/native/cpu/utils.h>
#include <ATen/ops/empty.h>

#include "kutacc.h"

#include "invariant_point.h"
#include "rigid.h"
#include "utils/memory.h"
#include "utils/linear.h"

namespace alphafold {

at::Tensor invariant_point_attention(at::Tensor &s, at::Tensor &z, at::Tensor &rigid_trans, at::Tensor &rigid_rot_mats,
                                     at::Tensor &mask, const InvariantPointAttentionWeight &weights)
{
    at::Tensor out = at::empty(s.sizes(), s.options());
    int64_t n_res = s.sizes()[0];
    int64_t c_s = weights.c_s;
    int64_t c_z = weights.c_z;
    int64_t c_hidden = weights.c_hidden;
    int64_t no_heads = weights.no_heads;
    int64_t no_qk_points = weights.no_qk_points;
    int64_t no_v_points = weights.no_v_points;

    KPEX_CHECK(s.scalar_type() == c10::kBFloat16, s.scalar_type());
    KPEX_CHECK(z.scalar_type() == c10::kBFloat16, z.scalar_type());
    KPEX_CHECK(rigid_trans.scalar_type() == c10::kFloat, rigid_trans.scalar_type());
    KPEX_CHECK(rigid_rot_mats.scalar_type() == c10::kFloat, rigid_rot_mats.scalar_type());
    KPEX_CHECK(mask.scalar_type() == c10::kBFloat16, mask.scalar_type());

    KPEX_CHECK_TENSOR_SHAPE(s, n_res, c_s);
    KPEX_CHECK_TENSOR_SHAPE(z, n_res, n_res, c_z);
    KPEX_CHECK_TENSOR_SHAPE(rigid_trans, n_res, 3);
    KPEX_CHECK_TENSOR_SHAPE(rigid_rot_mats, n_res, 3, 3);
    KPEX_CHECK_TENSOR_SHAPE(mask, n_res);

    rigid_trans = rigid_trans.view({n_res, 1, 1, 3});
    rigid_rot_mats = rigid_rot_mats.view({n_res, 1, 1, 3, 3});

    auto q = linear(s, weights.linear_q_w, weights.linear_q_b);
    auto k = linear(s, weights.linear_k_w, weights.linear_k_b);
    auto v = linear(s, weights.linear_v_w, weights.linear_v_b);

    auto q_pts = linear(s, weights.linear_q_points_w, weights.linear_q_points_b);
    q_pts = rigid_rot_vec_mul(q_pts, rigid_rot_mats, rigid_trans);

    auto k_pts = linear(s, weights.linear_k_points_w, weights.linear_k_points_b);
    k_pts = rigid_rot_vec_mul(k_pts, rigid_rot_mats, rigid_trans);
    auto v_pts = linear(s, weights.linear_v_points_w, weights.linear_v_points_b);
    v_pts = rigid_rot_vec_mul(v_pts, rigid_rot_mats, rigid_trans);
    v_pts = v_pts.permute({1, 2, 3, 0}).contiguous();

    auto b = at::empty({no_heads, n_res, n_res}, s.options());
    auto a = at::empty({no_heads, n_res, n_res}, q.options());
    auto head_weights = at::empty(weights.head_weights.sizes(), weights.head_weights.options());
    auto collect = at::empty({n_res, no_heads * (c_hidden + no_v_points * 4 + c_z)}, s.options());
    auto o = collect.narrow(1, 0, no_heads * c_hidden).view({n_res, no_heads, c_hidden});
    auto o_pt =
        collect.narrow(1, no_heads * c_hidden, no_heads * no_v_points * 3).view({n_res, 3, no_heads, no_v_points});
    auto o_pt_norm = collect.narrow(1, no_heads * (c_hidden + no_v_points * 3), no_heads * no_v_points)
                         .view({n_res, no_heads, no_v_points});
    auto o_pair =
        collect.narrow(1, no_heads * (c_hidden + no_v_points * 4), no_heads * c_z).view({n_res, no_heads, c_z});

    auto q_tw = convert_to_tensor_wrapper(q);
    auto k_tw = convert_to_tensor_wrapper(k);
    auto v_tw = convert_to_tensor_wrapper(v);
    auto q_pts_tw = convert_to_tensor_wrapper(q_pts);
    auto k_pts_tw = convert_to_tensor_wrapper(k_pts);
    auto v_pts_tw = convert_to_tensor_wrapper(v_pts);
    auto b_tw = convert_to_tensor_wrapper(b);
    auto a_tw = convert_to_tensor_wrapper(a);
    auto head_weights_tw = convert_to_tensor_wrapper(head_weights);
    auto weights_head_weights_tw = convert_to_tensor_wrapper(weights.head_weights);
    // auto collect_tw = convert_to_tensor_wrapper(collect);
    auto o_tw = convert_to_tensor_wrapper(o);
    auto o_pt_tw = convert_to_tensor_wrapper(o_pt);
    auto o_pt_norm_tw = convert_to_tensor_wrapper(o_pt_norm);
    auto o_pair_tw = convert_to_tensor_wrapper(o_pair);
    auto z_tw = convert_to_tensor_wrapper(z);
    auto rigid_rot_mats_tw = convert_to_tensor_wrapper(rigid_rot_mats);
    auto rigid_trans_tw = convert_to_tensor_wrapper(rigid_trans);
    auto mask_tw = convert_to_tensor_wrapper(mask);
    auto linear_b_w_tw = convert_to_tensor_wrapper(weights.linear_b_w);
    auto linear_b_b_tw = convert_to_tensor_wrapper(weights.linear_b_b);

    kutacc_af2_ipa_weights_t_wrapper *ipa_weight_ptr =
        new kutacc_af2_ipa_weights_t_wrapper(head_weights_tw, weights_head_weights_tw, linear_b_w_tw, linear_b_b_tw,
                                             c_z, c_hidden, no_heads, no_qk_points, no_v_points);
    kutacc_af2_ipa_s_inputs_t_wrapper *ipa_s_ptrs =
        new kutacc_af2_ipa_s_inputs_t_wrapper(a_tw, b_tw, q_tw, k_tw, v_tw, q_pts_tw, k_pts_tw, v_pts_tw, n_res);
    kutacc_af2_ipa_o_inputs_t_wrapper *ipa_o_ptrs =
        new kutacc_af2_ipa_o_inputs_t_wrapper(o_tw, o_pt_tw, o_pt_norm_tw, o_pair_tw);

    if (unlikely(ipa_s_ptrs == nullptr || ipa_o_ptrs == nullptr || ipa_weight_ptr == nullptr)) {
        return out;
    }

    kutacc_af2_invariant_point(ipa_s_ptrs, ipa_o_ptrs, z_tw.get_tensor(), rigid_rot_mats_tw.get_tensor(),
                               rigid_trans_tw.get_tensor(), mask_tw.get_tensor(), ipa_weight_ptr);

    out = linear(collect, weights.linear_out_w, weights.linear_out_b);
    delete ipa_weight_ptr;
    delete ipa_s_ptrs;
    delete ipa_o_ptrs;
    return out;
}

InvariantPointAttentionWeight::InvariantPointAttentionWeight(
    int64_t c_s, int64_t c_z, int64_t c_hidden, int64_t no_heads, int64_t no_qk_points, int64_t no_v_points,
    at::Tensor &linear_q_w, at::Tensor &linear_q_b, at::Tensor &linear_kv_w, at::Tensor linear_kv_b,
    at::Tensor &linear_q_points_w, at::Tensor &linear_q_points_b, at::Tensor &linear_kv_points_w,
    at::Tensor &linear_kv_points_b, at::Tensor &linear_b_w, at::Tensor &linear_b_b, at::Tensor &head_weights,
    at::Tensor &linear_out_w, at::Tensor &linear_out_b)
    : c_s(c_s), c_z(c_z), c_hidden(c_hidden), no_heads(no_heads), no_qk_points(no_qk_points), no_v_points(no_v_points)
{
    KPEX_CHECK(c_s > 0 && c_z > 0 && c_hidden > 0 && c_hidden <= INT32_MAX && no_heads > 0 && no_qk_points > 0 &&
                   no_v_points > 0 && no_qk_points <= 16 / 3 && no_v_points <= INT64_MAX - no_qk_points,
               "invalid int arg for InvariantPointAttentionWeight"); // no_qk_points is smaller than svcntw() / 3
    KPEX_CHECK(no_v_points <= INT64_MAX / 4 && c_z < INT64_MAX - 4 * no_v_points - c_hidden &&
                   no_heads < INT64_MAX / (c_hidden + no_v_points * 4 + c_z),
               "invalid shape for weights linear_out_w");
    KPEX_CHECK_TENSOR_SHAPE(linear_b_w, no_heads, c_z);
    KPEX_CHECK_TENSOR_SHAPE(linear_b_b, no_heads);
    KPEX_CHECK_TENSOR_SHAPE(head_weights, no_heads);
    KPEX_CHECK_TENSOR_SHAPE(linear_out_w, c_s, no_heads * (c_hidden + no_v_points * 4 + c_z));
    KPEX_CHECK_TENSOR_SHAPE(linear_out_b, c_s);
    // TO DO CHECK
    linear_q_w = linear_q_w.view({no_heads, c_hidden, c_s});
    linear_q_b = linear_q_b.view({no_heads, c_hidden});
    linear_kv_w = linear_kv_w.view({no_heads, 2 * c_hidden, c_s});
    linear_kv_b = linear_kv_b.view({no_heads, 2 * c_hidden});
    linear_q_points_w = linear_q_points_w.view({3, no_heads, no_qk_points, c_s}).permute({1, 2, 0, 3});
    linear_q_points_b = linear_q_points_b.view({3, no_heads, no_qk_points}).permute({1, 2, 0});
    linear_kv_points_w =
        linear_kv_points_w.view({3, no_heads, (no_qk_points + no_v_points), c_s}).permute({1, 2, 0, 3});
    linear_kv_points_b = linear_kv_points_b.view({3, no_heads, (no_qk_points + no_v_points)}).permute({1, 2, 0});

    auto float_opt = linear_q_w.options().device(kpex::device()).dtype(c10::kFloat);
    auto bf16_opt = linear_q_w.options().device(kpex::device()).dtype(c10::kBFloat16);
    this->linear_q_w = linear_q_w.to(bf16_opt).contiguous();
    this->linear_q_b = linear_q_b.to(float_opt).contiguous();
    this->linear_k_w = linear_kv_w.narrow(1, 0, c_hidden).to(bf16_opt).contiguous();
    this->linear_v_w = linear_kv_w.narrow(1, c_hidden, c_hidden).to(bf16_opt).contiguous();
    this->linear_k_b = linear_kv_b.narrow(1, 0, c_hidden).to(float_opt).contiguous();
    this->linear_v_b = linear_kv_b.narrow(1, c_hidden, c_hidden).to(float_opt).contiguous();
    this->linear_q_points_w = linear_q_points_w.to(bf16_opt).contiguous();
    this->linear_q_points_b = linear_q_points_b.to(float_opt).contiguous();
    this->linear_k_points_w = linear_kv_points_w.narrow(1, 0, no_qk_points).to(bf16_opt).contiguous();
    this->linear_k_points_b = linear_kv_points_b.narrow(1, 0, no_qk_points).to(float_opt).contiguous();
    this->linear_v_points_w = linear_kv_points_w.narrow(1, no_qk_points, no_v_points).to(bf16_opt).contiguous();
    this->linear_v_points_b = linear_kv_points_b.narrow(1, no_qk_points, no_v_points).to(float_opt).contiguous();
    this->linear_b_w = linear_b_w.to(bf16_opt).contiguous();
    this->linear_b_b = linear_b_b.to(float_opt).contiguous();
    this->head_weights = head_weights.to(float_opt).contiguous();
    this->linear_out_w = linear_out_w.to(bf16_opt).contiguous();
    this->linear_out_b = linear_out_b.to(float_opt).contiguous();
}
} // namespace alphafold
