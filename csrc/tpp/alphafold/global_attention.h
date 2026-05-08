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

#ifndef KPEX_TPP_ALPHAFOLD_GLOBAL_ATTENTION_H
#define KPEX_TPP_ALPHAFOLD_GLOBAL_ATTENTION_H

#include <ATen/core/Tensor.h>
#include "utils/check.h"

namespace alphafold {

struct GlobalAttentionWeight {
    int64_t nchannels;
    int64_t nheads;
    int64_t head_size;

    at::Tensor query_w;
    at::Tensor key_w;
    at::Tensor value_w;
    at::Tensor gating_w;
    at::Tensor gating_b;
    at::Tensor output_w;
    at::Tensor output_b;

    /**
         * @param query_w shape [nheads, head_size]
         * @param key_w shape [head_size, nchannels]
         * @param value_w shape [head_size, nchannels]
         * @param gating_w shape [nheads, head_size, nchannles]
         * @param gating_b shape [nheads, head_size]
         * @param output_w shape [nchannels, nheads, head_size]
         * @param output_b shape [nchannels]
        */
    GlobalAttentionWeight(at::Tensor &query_w, at::Tensor &key_w, at::Tensor &value_w, at::Tensor &gating_w,
                          at::Tensor &gating_b, at::Tensor &output_w, at::Tensor &output_b);
};

/**
     * @param q_data shape [batch, seq_len, nchannels]
     * @param m_data shape [batch, seq_len, nchannels]
     * @param q_mask shape [batch, seq_len, 1]
     */
at::Tensor global_attention(at::Tensor &q_data, at::Tensor &m_data, at::Tensor &q_mask,
                            const GlobalAttentionWeight &weights);

} // namespace alphafold

#endif
