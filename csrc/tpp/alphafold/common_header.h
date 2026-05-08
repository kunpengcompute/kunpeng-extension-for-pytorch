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

#ifndef KPEX_TPP_ALPHAFOLD_COMMON_HEADER_H
#define KPEX_TPP_ALPHAFOLD_COMMON_HEADER_H

#include "kutacc.h"

namespace alphafold {
struct kutacc_af2_attention_weights_t_wrapper : kutacc_af2_attention_weights_t {
    kutacc_af2_attention_weights_t_wrapper(kutacc::TensorWrapper &query_w, kutacc::TensorWrapper &key_w,
                                           kutacc::TensorWrapper &value_w, kutacc::TensorWrapper &gating_w,
                                           kutacc::TensorWrapper &gating_b, kutacc::TensorWrapper &output_w,
                                           kutacc::TensorWrapper &output_b, int64_t nchannels, int64_t nheads,
                                           int64_t head_size)
    {
        this->nchannels = nchannels;
        this->nheads = nheads;
        this->head_size = head_size;
        this->query_w = query_w.get_tensor();
        this->key_w = key_w.get_tensor();
        this->value_w = value_w.get_tensor();
        this->gating_w = gating_w.get_tensor();
        this->gating_b = gating_b.get_tensor();
        this->output_w = output_w.get_tensor();
        this->output_b = output_b.get_tensor();
    }
};

struct kutacc_af2_attention_inputs_t_wrapper : kutacc_af2_attention_inputs_t {
    kutacc_af2_attention_inputs_t_wrapper(kutacc::TensorWrapper &q, kutacc::TensorWrapper &k, kutacc::TensorWrapper &v,
                                          kutacc::TensorWrapper &gate, kutacc::TensorWrapper &avg, int64_t batch,
                                          int64_t seq_len)
    {
        this->batch = batch;
        this->seq_len = seq_len;
        this->q = q.get_tensor();
        this->k = k.get_tensor();
        this->v = v.get_tensor();
        this->gate = gate.get_tensor();
        this->avg = avg.get_tensor();
    }
};
} // namespace alphafold
#endif
