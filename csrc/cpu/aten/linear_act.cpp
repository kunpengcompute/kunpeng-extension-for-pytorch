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

#include "linear_act.h"
#include <torch/all.h>
#include <ATen/native/Resize.h>
#include <cstdlib>

namespace kpex {
// Actual implementation for linear_act,linear_act.out variations (linear+activation fused kernel). Returns activation_function(m1@m2.t+bias) to result tensor
static void linear_act_out_impl_(at::Tensor &result, const at::Tensor &m1, const at::Tensor &m2, const at::Tensor &bias) {
    RECORD_FUNCTION("KPEX::linear_act_out_impl_", std::vector<c10::IValue>{m1, m2});

    const auto input_dim = m1.dim();
    const auto weight_dim = m2.dim();
    TORCH_CHECK(input_dim != 0 ,
                "kpex: input needs to be at least 1D, but it is ",
                input_dim, "D ");

    TORCH_INTERNAL_ASSERT(weight_dim == 2);

    TORCH_CHECK(
        m1.dtype() == m2.dtype(),
        "kpex: expected m1 and m2 to have the same dtype, but got: ", m1.dtype(), " != ", m2.dtype()
    )
    
    // Array access is faster than .size(n) and .stride(n)
    auto m1_inner = m1.size(-1);
    auto m2_sizes = m2.sizes();

    TORCH_CHECK(
        m1_inner == m2_sizes[1], // [out_features, in_features]
        "kpex: input shape is incompatible with matrix multiplication (",
        m1_inner, "!=", m2_sizes[1], ")");

    // resulting matrix has zero elements
    if (result.numel() == 0) {
        return;
    }
    // Inner Dimension = 0
    if (m1_inner == 0) {
        return;
    }

    TORCH_CHECK(bias.defined(), "KPEX does not support linear_act without bias.");

    // Setup and call KuDNN LinearActivationLayer

    // Pytorch handles initialization and update of number of work threads
    int numThreads = at::get_num_threads();
    // Currently supports 2D. Support for nDim will be added in future version
    KuDNN::Layout srcLayout; // Input can be nDim [1 to n]
    at::Tensor a;
    if (m1.is_contiguous()) {
            srcLayout = KuDNN::Layout::ROW_MAJOR; // supports 2D
            a = m1;
    }
    else if (m1.t().is_contiguous()) {
            srcLayout = KuDNN::Layout::COL_MAJOR; // supports 2D
            a = m1;
    }
    else {
        srcLayout = KuDNN::Layout::ROW_MAJOR; // supports 2D
        a = m1.clone(at::MemoryFormat::Contiguous);
    }

    KuDNN::Layout weiLayout; // [out_features, in_features] so always COL_MAJOR for BLAS
    at::Tensor b;
    weiLayout = KuDNN::Layout::COL_MAJOR;
    b = m2;

    KuDNN::Layout dstLayout;
    at::Tensor c;
    if (result.is_contiguous()) {
            dstLayout = KuDNN::Layout::ROW_MAJOR; // supports 2D
            c = result;
    }
    else if (result.t().is_contiguous()) {
            dstLayout = KuDNN::Layout::COL_MAJOR; // supports 2D
            c = result;
    }
    else {
        dstLayout = KuDNN::Layout::ROW_MAJOR; // supports 2D
        c = result.clone(at::MemoryFormat::Contiguous);
    }

    auto a_sizes = a.sizes();
    auto b_sizes = b.sizes();
    int M = a_sizes[0]; // supports 2D
    int K = b_sizes[1]; // [out_features, in_features]
    int N = b_sizes[0]; // [out_features, in_features]

    KuDNN::Shape srcShape(M, K); // supports 2D
    KuDNN::Shape weiShape(K, N); // This is how KuDNN/KBLAS gets the shape
    KuDNN::Shape dstShape(M, N); // supports 2D

    KuDNN::TensorInfo srcTensor{srcShape, KuDNN::Element::MatchType<__bf16>(), srcLayout};
    KuDNN::TensorInfo weiTensor{weiShape, KuDNN::Element::MatchType<__bf16>(), weiLayout};
    KuDNN::TensorInfo dstTensor{dstShape, KuDNN::Element::MatchType<__bf16>(), dstLayout};

    DISPATCH_TO_CPP_TYPES(a.scalar_type(), "linear_act_out_impl_", {
        // Inside this lambda, 'cpp_type' is the C++ type corresponding to self.scalar_type()
        using a_cpp_type = cpp_type;
        srcTensor = {srcShape, KuDNN::Element::MatchType<a_cpp_type>(), srcLayout};
    });
    DISPATCH_TO_CPP_TYPES(b.scalar_type(), "linear_act_out_impl_", {
        using b_cpp_type = cpp_type;
        weiTensor = {weiShape, KuDNN::Element::MatchType<b_cpp_type>(), weiLayout};
    });
    DISPATCH_TO_CPP_TYPES(c.scalar_type(), "linear_act_out_impl_", {
        using c_cpp_type = cpp_type;
        dstTensor = {dstShape, KuDNN::Element::MatchType<c_cpp_type>(), dstLayout};
    });

    float alpha_f = 1.0f; // linear operation is (self @ weight.t + bias) => gemm is (result = alpha*(self @ weight.t) + beta*result + bias)
    float beta_f = 0.0f; // KuDNN LinearActivationLayerFWD uses beta=0 to run gemm, so no in-place operation
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH; // Supports SILU (=SWISH) activation function. Will be generalized to any activation function supported by KuDNN

    if (bias.defined()) {
        KuDNN::Layout biasLayout;
        at::Tensor d;
        if (bias.is_contiguous()) {
                biasLayout = KuDNN::Layout::ROW_MAJOR; // supports 2D
                d = bias;
        }
        else if (bias.t().is_contiguous()) { // supports 2D
                biasLayout = KuDNN::Layout::COL_MAJOR;
                d = bias;
        }
        else {
            biasLayout = KuDNN::Layout::ROW_MAJOR; // supports 2D
            d = bias.clone(at::MemoryFormat::Contiguous);
        }
        KuDNN::Shape biasShape(M, N); // supports 2D
        KuDNN::TensorInfo biasTensor{biasShape, KuDNN::Element::MatchType<__bf16>(), biasLayout};
        DISPATCH_TO_CPP_TYPES(d.scalar_type(), "linear_act_out_impl_", {
            using c_cpp_type = cpp_type;
            biasTensor = {biasShape, KuDNN::Element::MatchType<c_cpp_type>(), biasLayout};
        });

        KuDNN::LinearActivationLayerFWD linearActivationLayerFwd(srcTensor, weiTensor, dstTensor, biasTensor, alpha_f, beta_f, algKind, numThreads);
        linearActivationLayerFwd.Run(a.data_ptr(), b.data_ptr(), c.data_ptr(), d.data_ptr(), numThreads);
    } else {
        // TODO: KuDNN LinearActivationLayerFWD has no implementation for "Run" without bias
        KuDNN::LinearActivationLayerFWD linearActivationLayerFwd(srcTensor, weiTensor, dstTensor, alpha_f, beta_f, algKind, numThreads);
        // linearActivationLayerFwd.Run(a.data_ptr(), b.data_ptr(), c.data_ptr(), numThreads);
    }

    // Check if c and result are the same object, before returning result
    if (!c.is_same(result)) {
        result.copy_(c);
    }
}
// Wrapper for linear_act.out variation (linear+activation fused kernel) implementation. Returns activation_function(self@weight.t+bias_opt) to result tensor
at::Tensor& linear_act_out_impl(const at::Tensor& self, const at::Tensor& weight, const std::optional<at::Tensor>& bias_opt, at::Tensor &result) {
    RECORD_FUNCTION("KPEX::linear_act_out_impl", std::vector<c10::IValue>{self, weight});

    // Set output shape as [all batch sizes, in_features]
    auto output_shape = self.sizes().vec();
    output_shape.back() = weight.size(0); // [out_features, in_features]
    at::native::resize_output(result, output_shape);
    // Handle optional tensor type
    auto bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const at::Tensor& bias_ref = *bias_maybe_owned;
    if (bias_ref.defined())
    {
        auto b_bias = bias_ref.expand(result.sizes()); // expand if broadcastable
        at::NoNamesGuard guard; // Disable Named Tensor checks (metadata) inside scope; treat tensors as standard positional arrays
        linear_act_out_impl_(result, self, weight, b_bias);
    } else {
        at::NoNamesGuard guard; // Disable Named Tensor checks (metadata) inside scope; treat tensors as standard positional arrays
        linear_act_out_impl_(result, self, weight, bias_ref);
    }
    return result;
}
// Wrapper for linear_act (linear+activation fused kernel) implementation. Allocates result tensor and returns activation_function(self@weight.t+bias_opt)
at::Tensor linear_act_impl(const at::Tensor& self, const at::Tensor& weight, const std::optional<at::Tensor>& bias_opt) {
    RECORD_FUNCTION("KPEX::linear_act_impl", std::vector<c10::IValue>{self, weight});
    
    // Set output shape as [all batch sizes, in_features]
    auto output_shape = self.sizes().vec();
    output_shape.back() = weight.size(0); // [out_features, in_features]
    at::Tensor output = at::empty(output_shape, weight.options());
    output = linear_act_out_impl(self, weight, bias_opt, output);
    return output;
}

}   // namespace kpex


namespace {
// Graph Optimization/Fusion for custom operators
// Register in a custom namespace
TORCH_LIBRARY_FRAGMENT(kpex, m) {
    m.def("linear_act(Tensor self, Tensor weight, Tensor? bias=None) -> Tensor");
    m.def("linear_act.out(Tensor self, Tensor weight, Tensor? bias=None, *, Tensor(a!) out) -> Tensor(a!)");
}
TORCH_LIBRARY_IMPL(kpex, CPU, m) {
    m.impl( TORCH_SELECTIVE_NAME("kpex::linear_act"), TORCH_FN((&kpex::linear_act_impl)) );
    m.impl( TORCH_SELECTIVE_NAME("kpex::linear_act.out"), TORCH_FN((&kpex::linear_act_out_impl)) );
}

} // namespace