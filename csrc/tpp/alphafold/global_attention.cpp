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

#include "kutacc.h"
#include "global_attention.h"
#include "common_header.h"
#include "utils/memory.h"
#include "utils/TensorWrapper.h"

namespace alphafold {

    GlobalAttentionWeight::GlobalAttentionWeight(at::Tensor &query_w, at::Tensor &key_w, at::Tensor &value_w, at::Tensor &gating_w, 
        at::Tensor &gating_b, at::Tensor &output_w, at::Tensor &output_b)
    {
        KPEX_CHECK(query_w.dim() == 3, "invalid tensor shape, query_w must have 3 dims");
        int64_t nchannels = query_w.sizes()[2];
        int64_t nheads = query_w.sizes()[0];
        int64_t head_size = query_w.sizes()[1];
        KPEX_CHECK(nchannels > 0 && nheads > 0 && head_size > 0 && nchannels <= INT32_MAX && head_size <= INT32_MAX && nheads <= INT32_MAX
            && nchannels == nheads * head_size, "invalid query_w shape [", nchannels, ", ", nheads, ", ", head_size, "]");
        this->nchannels = nchannels;
        this->nheads = nheads;
        this->head_size = head_size;
        KPEX_CHECK_TENSOR_SHAPE(key_w, head_size, nchannels);
        KPEX_CHECK_TENSOR_SHAPE(value_w, head_size, nchannels);
        KPEX_CHECK_TENSOR_SHAPE(gating_w, nheads, head_size, nchannels);
        KPEX_CHECK_TENSOR_SHAPE(gating_b, nheads, head_size);
        KPEX_CHECK_TENSOR_SHAPE(output_w, nchannels, nheads, head_size);
        KPEX_CHECK_TENSOR_SHAPE(output_b, nchannels);

        auto float_opt = query_w.options().device(kpex::device()).dtype(c10::kFloat);
        auto bf16_opt = query_w.options().device(kpex::device()).dtype(c10::kBFloat16);

        query_w = query_w.to(bf16_opt).contiguous().view({nchannels, nchannels});
        key_w = key_w.to(bf16_opt).contiguous().view({head_size, nchannels});
        value_w = value_w.to(bf16_opt).contiguous().view({head_size, nchannels});
        gating_w = gating_w.to(bf16_opt).contiguous().view({nchannels, nchannels});
        output_w = output_w.to(bf16_opt).contiguous().view({nchannels, nchannels});

        this->query_w = linear_weight_prepack(query_w);
        this->key_w = linear_weight_prepack(key_w);
        this->value_w = linear_weight_prepack(value_w);
        this->gating_w = linear_weight_prepack(gating_w);
        this->output_w = linear_weight_prepack(output_w);

        this->gating_b = gating_b.to(float_opt).contiguous();
        this->output_b = output_b.to(float_opt).contiguous();
    }

    at::Tensor global_attention(at::Tensor &q_data, at::Tensor &m_data, at::Tensor &q_mask, const GlobalAttentionWeight &weights)
    {
        at::Tensor out = at::empty(q_data.sizes(), q_data.options());
        int64_t batch = q_data.sizes()[0];
        int64_t seq_len = q_data.sizes()[1];
        int64_t nchannels = weights.nchannels;
        int64_t nheads = weights.nheads;
        int64_t head_size = weights.head_size;

        KPEX_CHECK(q_data.dtype() == c10::kBFloat16, q_data.dtype());
        KPEX_CHECK(m_data.dtype() == c10::kBFloat16, m_data.dtype());
        KPEX_CHECK(q_mask.dtype() == c10::kBFloat16, q_mask.dtype());
        KPEX_CHECK_TENSOR_SHAPE(q_data, batch, seq_len, nchannels);
        KPEX_CHECK_TENSOR_SHAPE(q_mask, batch, seq_len, 1);

        q_mask = q_mask.contiguous();

        auto q_avg = q_data.new_empty({batch, nchannels});
        auto q = q_data.new_empty({batch, nheads, head_size});
        auto k = q_data.new_empty({batch, seq_len, head_size});
        auto v = q_data.new_empty({head_size, batch, seq_len});
        auto gate = q_data.new_empty({batch, seq_len, nheads, head_size});

        auto q_avg_tw = convert_to_tensor_wrapper(q_avg);
        auto q_tw = convert_to_tensor_wrapper(q);
        auto k_tw = convert_to_tensor_wrapper(k);
        auto v_tw = convert_to_tensor_wrapper(v);
        auto gate_tw = convert_to_tensor_wrapper(gate);
        auto q_data_tw = convert_to_tensor_wrapper(q_data);
        auto q_mask_tw = convert_to_tensor_wrapper(q_mask);
        auto out_tw = convert_to_tensor_wrapper(out);
        auto query_w_tw = convert_to_tensor_wrapper(weights.query_w);
        auto key_w_tw = convert_to_tensor_wrapper(weights.key_w);
        auto value_w_tw = convert_to_tensor_wrapper(weights.value_w);
        auto gating_w_tw = convert_to_tensor_wrapper(weights.gating_w);
        auto gating_b_tw = convert_to_tensor_wrapper(weights.gating_b);
        auto output_w_tw = convert_to_tensor_wrapper(weights.output_w);
        auto output_b_tw = convert_to_tensor_wrapper(weights.output_b);

        kutacc_af2_attention_weights_t_wrapper *global_attention_weight_ptr = new kutacc_af2_attention_weights_t_wrapper(query_w_tw, key_w_tw, value_w_tw, gating_w_tw, gating_b_tw,
        output_w_tw, output_b_tw, nchannels, nheads, head_size);

        kutacc_af2_attention_inputs_t_wrapper *global_attention_q_ptr = new kutacc_af2_attention_inputs_t_wrapper(q_tw, k_tw, v_tw, gate_tw, q_avg_tw, batch, seq_len);
        if (unlikely(global_attention_weight_ptr == nullptr || global_attention_q_ptr == nullptr)) {
            return out;
        }
        kutacc_af2_global_attention(global_attention_q_ptr, q_data_tw.get_tensor(), q_mask_tw.get_tensor(), global_attention_weight_ptr, out_tw.get_tensor());
        delete global_attention_weight_ptr;
        delete global_attention_q_ptr;
        return out;
    }
}