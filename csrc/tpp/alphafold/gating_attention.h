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

#ifndef KPEX_TPP_ALPHAFOLD_GATING_ATTENTION_H
#define KPEX_TPP_ALPHAFOLD_GATING_ATTENTION_H

#include <ATen/core/Tensor.h>
#include "utils/check.h"

namespace alphafold{
struct GatingAttentionWeight {
    int64_t nchannels;
    int64_t nheads;
    int64_t head_size;

    at::Tensor query_w;     //dtype=bf16
    at::Tensor key_w;       //dtype=bf16
    at::Tensor value_w;     //dtype=bf16
    at::Tensor gating_w;    //dtype=bf16
    at::Tensor gating_b;    //dtype=bf16
    at::Tensor output_w;    //dtype=bf16
    at::Tensor output_b;    //dtype=bf16

    /**
     * @param query_w shape [nheads, head_size, nchannels], dtype=any
     * @param key_w shape [nheads, head_size, nchannels], dtype=any
     * @param value_w shape [nheads, head_size, nchannels], dtype=any
     * @param gating_w shape [nchannels, nheads, head_size], dtype=any
     * @param gating_b shape [nheads, head_size], dtype=any
     * @param output_w shape [nchannels, nheads, head_size], dtype=any
     * @param output_b shape [nchannels], dtype=any
     */
    GatingAttentionWeight (at::Tensor &query_w, at::Tensor &key_w, at::Tensor &value_w, at::Tensor &gating_w,
        at::Tensor &gating_b, at::Tensor &output_w, at::Tensor &output_b);
};

/**
 * @param q_data shape [batch, seq_len, nchannels], bf16
 * @param m_data shape [batch, seq_len, nchannels], bf16
 * @param bias shape [batch, 1, 1, seq_len], bf16
 * @param nonbatched_bias shape [nheads, seq_len, seq_len] or [0], bf16
 */
at::Tensor gating_attention(at::Tensor &q_data, at::Tensor &m_data, at::Tensor &bias, at::Tensor &nonbatched_bias,
    const GatingAttentionWeight &weights, std::optional<int64_t> block_size);
}   // namespace alphafold

#endif
