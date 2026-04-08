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

#include "kudnn_adapter.h"
#include <cstdlib>
#include <stdexcept>

#define MICRO_MATRIX_OPS 5e6
#define MIN_MATRIX_OPS 1e8
#define MIN_TO_MID_MATRIX_OPS 1e9
#define MID_MATRIX_OPS 1e10
#define LARGE_MATRIX_OPS 5e11

namespace kudnn {
inline KuDNN::Element::TypeT get_kudnn_element_type(at::ScalarType scalar_type) {
    static const std::unordered_map<at::ScalarType, KuDNN::Element::TypeT> linear_type_map = {
        {at::kFloat, KuDNN::Element::TypeT::F32},
        {at::kHalf, KuDNN::Element::TypeT::F16},
    };

    auto it = linear_type_map.find(scalar_type);
    if (it == linear_type_map.end()) {
        throw std::invalid_argument("Unsupported scalar type for kudnn linear operation");
    }
    return it->second;
}

inline KuDNN::TensorInfo get_kudnn_linear_tensor(const at::Tensor& tensor) {
    KuDNN::Shape shape(tensor.size(0), tensor.size(1));
    KuDNN::Element::TypeT type = get_kudnn_element_type(tensor.scalar_type());
    KuDNN::Layout layout = KuDNN::Layout::ROW_MAJOR;
    return KuDNN::TensorInfo(shape, type, layout);
}

// y = x @ weight^T + bias
at::Tensor kudnn_linear(const at::Tensor& input, const at::Tensor& weight, const std::optional<at::Tensor>& bias) {
    KPEX_CHECK(isValidateTensor(input), "input is not validate, unsupported");
    KPEX_CHECK(weight.dim() <= 2, "weight has more than 2 dimensions, unsupported");
    const auto input_dim = input.dim();

    auto input_reshaped = (input_dim == 2) 
        ? input
        : input.view({-1, input.size(input_dim - 1)});
    const int64_t m = input_reshaped.size(0);
    const int64_t k = input_reshaped.size(1); 
    const int64_t n = weight.size(0);      
    const uint64_t total_ops = bias.has_value() ? (2ULL * k + 1ULL) * m * n : 2ULL * m * k * n; // total ops ≈ 2×m×k×n

    int num_threads = 1;
    if (total_ops > MICRO_MATRIX_OPS && total_ops <= MIN_MATRIX_OPS) {
        num_threads = 4; // small size matrix (num of ops less than 100 m):using 4 thread to avoid overhead of multi-threading
    } else if (total_ops > MIN_MATRIX_OPS && total_ops <= MIN_TO_MID_MATRIX_OPS) {
        num_threads = 9; // num of ops less than 1 billion, use 9 threads to enhance parallel performance
    } else if (total_ops > MIN_TO_MID_MATRIX_OPS && total_ops <= MID_MATRIX_OPS) {
        num_threads = 18; // medium matrix size (ops is bigger than 1 billion, less than 10 billion), computional density increase
    } else if (total_ops > MID_MATRIX_OPS && total_ops <= LARGE_MATRIX_OPS) {
        num_threads = 27; // computional density increase more
    } else if (total_ops > LARGE_MATRIX_OPS) {
        num_threads = 36; // max computional density for a numa
    }

    c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias);
    const at::Tensor& bias_new = *bias_maybe_owned;

    const std::vector<int64_t> out_sizes_vec = {input_reshaped.size(0), weight.size(0)};

    auto input_opt = at::TensorOptions().device(input_reshaped.device()).dtype(input_reshaped.scalar_type());
    at::Tensor out = at::empty(out_sizes_vec, input_opt);

    KuDNN::TensorInfo input_tensorInfo = get_kudnn_linear_tensor(input_reshaped);
    const auto weight_kudnn_type = get_kudnn_element_type(weight.scalar_type());
    KuDNN::Shape weight_shape(weight.size(1), weight.size(0));
    KuDNN::TensorInfo weight_tensorInfo = {weight_shape, weight_kudnn_type, KuDNN::Layout::BA};

    KuDNN::TensorInfo out_tensorInfo = get_kudnn_linear_tensor(out);

    if (bias_new.defined()) {
        const auto bias_kudnn_type = get_kudnn_element_type(bias_new.scalar_type());
        KuDNN::Shape bias_shape(1, bias_new.size(0));
        KuDNN::Shape bias_stride(0, 1);
        KuDNN::TensorInfo bias_tensorInfo = {bias_shape, bias_kudnn_type, KuDNN::Layout::AB, bias_stride};

        KuDNN::Gemm gemmLayer(input_tensorInfo, weight_tensorInfo, out_tensorInfo, bias_tensorInfo, num_threads);
        gemmLayer.Run(input_reshaped.data_ptr(), weight.data_ptr(), out.data_ptr(), bias_new.data_ptr(), 1.0f, 0.0f, num_threads);
    } else {
        KuDNN::Gemm gemmLayer(input_tensorInfo, weight_tensorInfo, out_tensorInfo, num_threads);
        gemmLayer.Run(input_reshaped.data_ptr(), weight.data_ptr(), out.data_ptr(), 1.0f, 0.0f, num_threads);
    }

    if (input_dim != 2) {
        std::vector<int64_t> output_size;
        output_size.reserve(input_dim);
        for (int i = 0; i < input_dim - 1; ++i) {
            output_size.push_back(input.size(i));
        }
        output_size.push_back(weight.size(0));
        return out.reshape(output_size);
    } else {
        return out;
    }
}

} // namespace kudnn