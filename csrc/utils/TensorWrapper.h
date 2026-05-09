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

#ifndef KPEX_UTILS_TENSORWRAPPER_H
#define KPEX_UTILS_TENSORWRAPPER_H

#include <vector>
#include <ATen/core/Tensor.h>
#include <memory.h>
#include "kutacc.h"

inline kutacc::TensorWrapper convert_to_tensor_wrapper(at::Tensor &tensor)
{
    kutacc::DType dtype = kutacc::kInt64;
    auto scalar_type = tensor.scalar_type();
    if (scalar_type == at::kBFloat16) {
        dtype = kutacc::kBF16;
    }
    return kutacc::TensorWrapper(tensor.data_ptr(), tensor.sizes().vec(), tensor.strides().vec(), tensor.dim(), dtype);
}

inline const kutacc::TensorWrapper convert_to_tensor_wrapper(const at::Tensor &tensor)
{
    kutacc::DType dtype = kutacc::kInt64;
    auto scalar_type = tensor.scalar_type();
    if (scalar_type == at::kBFloat16) {
        dtype = kutacc::kBF16;
    }
    return kutacc::TensorWrapper(tensor.data_ptr(), tensor.sizes().vec(), tensor.strides().vec(), tensor.dim(), dtype);
}

inline kutacc::TensorWrapper convert_to_tensor_wrapper_comm(at::Tensor &tensor)
{
    int64_t scalar_size = c10::elementSize(tensor.scalar_type());
    kutacc::DType dtype = kutacc::kInt64;
    auto scalar_type = tensor.scalar_type();
    if (scalar_type == at::kBFloat16) {
        dtype = kutacc::kBF16;
    }
    return kutacc::TensorWrapper(
        tensor.data_ptr(), {tensor.sizes()[0], tensor.sizes()[1], tensor.sizes()[2] * scalar_size},
        {tensor.strides()[0] * scalar_size, tensor.strides()[1] * scalar_size, 1}, tensor.dim(), dtype);
}

inline const kutacc::TensorWrapper convert_to_tensor_wrapper_comm(const at::Tensor &tensor)
{
    int64_t scalar_size = c10::elementSize(tensor.scalar_type());
    kutacc::DType dtype = kutacc::kInt64;
    auto scalar_type = tensor.scalar_type();
    if (scalar_type == at::kBFloat16) {
        dtype = kutacc::kBF16;
    }
    return kutacc::TensorWrapper(
        tensor.data_ptr(), {tensor.sizes()[0], tensor.sizes()[1], tensor.sizes()[2] * scalar_size},
        {tensor.strides()[0] * scalar_size, tensor.strides()[1] * scalar_size, 1}, tensor.dim(), dtype);
}

inline at::Tensor linear_weight_prepack(const at::Tensor &weight, int64_t num_threads = 0)
{
    int64_t n = weight.sizes()[0];
    int64_t k = weight.sizes()[1];
    int64_t ldb = weight.strides()[0];
    int64_t pack_size = kutacc_af2_gemm_pack_get_size('A', 'T', 'N', n, 0, k);

    at::Tensor result = weight.new_empty({pack_size});
    kutacc_af2_linear_weight_prepack((__bf16 *)weight.data_ptr(), (__bf16 *)result.data_ptr(), n, k, ldb, num_threads);
    return result;
}

#endif
