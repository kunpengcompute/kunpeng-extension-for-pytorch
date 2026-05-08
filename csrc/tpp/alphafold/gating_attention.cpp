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
#include <ATen/record_function.h>

#include "gating_attention.h"
#include "kutacc.h"
#include "common_header.h"
#include "utils/memory.h"
#include "utils/TensorWrapper.h"

namespace alphafold {
namespace {
int64_t default_block_size(int64_t seq_len)
{
    if (seq_len < 300) {
        return 176;
    } else if (seq_len < 600) {
        return 128;
    } else if (seq_len < 800) {
        return 80;
    } else if (seq_len < 1300) {
        return 64;
    } else if (seq_len < 1700) {
        return 48;
    } else {
        return 32;
    }
}
} // namespace

GatingAttentionWeight::GatingAttentionWeight(at::Tensor &query_w, at::Tensor &key_w, at::Tensor &value_w,
                                             at::Tensor &gating_w, at::Tensor &gating_b, at::Tensor &output_w,
                                             at::Tensor &output_b)
{
    KPEX_CHECK(query_w.dim() == 3, "query_w's dim must be 3 dims");
    int64_t nchannels = query_w.sizes()[2];
    int64_t nheads = query_w.sizes()[0];
    int64_t head_size = query_w.sizes()[1];
    KPEX_CHECK(nchannels > 0 && nheads > 0 && head_size > 0 && nchannels <= INT32_MAX && head_size <= INT32_MAX &&
                   nheads <= INT32_MAX && nchannels == nheads * head_size,
               "invalid query_w shape [", nchannels, ", ", nheads, ", ", head_size, "]");
    KPEX_CHECK_TENSOR_SHAPE(key_w, nheads, head_size, nchannels);
    KPEX_CHECK_TENSOR_SHAPE(value_w, nheads, head_size, nchannels);
    KPEX_CHECK_TENSOR_SHAPE(gating_w, nheads, head_size, nchannels);
    KPEX_CHECK_TENSOR_SHAPE(gating_b, nheads, head_size);
    KPEX_CHECK_TENSOR_SHAPE(output_w, nchannels, nheads, head_size);
    KPEX_CHECK_TENSOR_SHAPE(output_b, nchannels);
    this->nchannels = nchannels;
    this->nheads = nheads;
    this->head_size = head_size;

    auto float_opt = query_w.options().device(kpex::device()).dtype(c10::kFloat);
    auto bf16_opt = query_w.options().device(kpex::device()).dtype(c10::kBFloat16);
    query_w = query_w.to(bf16_opt).contiguous().view({nchannels, nchannels});
    key_w = key_w.to(bf16_opt).contiguous().view({nchannels, nchannels});
    value_w = value_w.to(bf16_opt).contiguous().view({nchannels, nchannels});
    gating_w = gating_w.to(bf16_opt).contiguous().view({nchannels, nchannels});
    output_w = output_w.to(bf16_opt).contiguous().view({nchannels, nchannels});

    auto query_w_res = linear_weight_prepack(query_w);
    auto key_w_res = linear_weight_prepack(key_w);
    auto value_w_res = linear_weight_prepack(value_w);
    auto gating_w_res = linear_weight_prepack(gating_w);
    auto output_w_res = linear_weight_prepack(output_w);

    this->query_w = query_w_res;
    this->key_w = key_w_res;
    this->value_w = value_w_res;
    this->gating_w = gating_w_res;
    this->output_w = output_w_res;

    this->gating_b = gating_b.to(float_opt).contiguous();
    this->output_b = output_b.to(float_opt).contiguous();
}

at::Tensor gating_attention(at::Tensor &q_data, at::Tensor &m_data, at::Tensor &bias, at::Tensor &nonbatched_bias,
                            const GatingAttentionWeight &weights, std::optional<int64_t> block_size)
{
    at::Tensor out = at::empty(q_data.sizes(), q_data.options());
    int64_t batch = q_data.sizes()[0];
    int64_t seq_len = q_data.sizes()[1];
    int64_t nchannels = weights.nchannels;
    int64_t nheads = weights.nheads;
    int64_t head_size = weights.head_size;
    int64_t block_size_ = default_block_size(seq_len);
    block_size_ = block_size.value_or(block_size_);

    RECORD_FUNCTION("gating_attention", c10::ArrayRef<c10::IValue>({batch, seq_len, nheads, head_size}));
    KPEX_CHECK(q_data.dtype() == c10::kBFloat16, q_data.dtype());
    KPEX_CHECK(m_data.dtype() == c10::kBFloat16, m_data.dtype());
    KPEX_CHECK(bias.dtype() == c10::kBFloat16, bias.dtype());
    KPEX_CHECK(nonbatched_bias.dtype() == c10::kBFloat16, bias.dtype());
    KPEX_CHECK_TENSOR_SHAPE(q_data, batch, seq_len, nchannels);
    KPEX_CHECK_TENSOR_SHAPE(bias, batch, 1, 1, seq_len);
    if (nonbatched_bias.sizes()[0] != 0) {
        KPEX_CHECK_TENSOR_SHAPE(nonbatched_bias, nheads, seq_len, seq_len);
    }

    bias = bias.contiguous();
    nonbatched_bias = nonbatched_bias.contiguous();

    auto q = q_data.new_empty({batch, seq_len, nheads, head_size});
    auto k = q_data.new_empty({batch, seq_len, nheads, head_size});
    auto v = q_data.new_empty({nheads, head_size, batch, seq_len});
    auto gate = q_data.new_empty({batch, seq_len, nheads, head_size});
    auto weighted_avg = q_data.new_empty({batch, seq_len, nheads, head_size});
    at::Tensor input;
    {
        RECORD_FUNCTION("input_prepack", c10::ArrayRef<c10::IValue>({}));
        input = linear_weight_prepack(q_data.view({batch * seq_len, nchannels}));
    }

    auto input_tw = convert_to_tensor_wrapper(input);
    auto q_tw = convert_to_tensor_wrapper(q);
    auto k_tw = convert_to_tensor_wrapper(k);
    auto v_tw = convert_to_tensor_wrapper(v);
    auto gate_tw = convert_to_tensor_wrapper(gate);
    auto weighted_avg_tw = convert_to_tensor_wrapper(weighted_avg);
    auto bias_tw = convert_to_tensor_wrapper(bias);
    auto nonbatched_bias_tw = convert_to_tensor_wrapper(nonbatched_bias);
    auto query_w_tw = convert_to_tensor_wrapper(weights.query_w);
    auto key_w_tw = convert_to_tensor_wrapper(weights.key_w);
    auto value_w_tw = convert_to_tensor_wrapper(weights.value_w);
    auto gating_w_tw = convert_to_tensor_wrapper(weights.gating_w);
    auto gating_b_tw = convert_to_tensor_wrapper(weights.gating_b);
    auto output_w_tw = convert_to_tensor_wrapper(weights.output_w);
    auto output_b_tw = convert_to_tensor_wrapper(weights.output_b);
    auto out_tw = convert_to_tensor_wrapper(out);

    kutacc_af2_attention_weights_t_wrapper *gating_attention_weight_ptr =
        new kutacc_af2_attention_weights_t_wrapper(query_w_tw, key_w_tw, value_w_tw, gating_w_tw, gating_b_tw,
                                                   output_w_tw, output_b_tw, nchannels, nheads, head_size);
    kutacc_af2_attention_inputs_t_wrapper *gating_attention_q_ptr =
        new kutacc_af2_attention_inputs_t_wrapper(q_tw, k_tw, v_tw, gate_tw, weighted_avg_tw, batch, seq_len);

    if (unlikely(gating_attention_weight_ptr == nullptr || gating_attention_q_ptr == nullptr)) {
        return out;
    }

    kutacc_af2_gating_attention(input_tw.get_tensor(), gating_attention_q_ptr, bias_tw.get_tensor(),
                                nonbatched_bias_tw.get_tensor(), gating_attention_weight_ptr, out_tw.get_tensor(),
                                block_size_);
    delete gating_attention_weight_ptr;
    delete gating_attention_q_ptr;
    return out;
}
} // namespace alphafold
