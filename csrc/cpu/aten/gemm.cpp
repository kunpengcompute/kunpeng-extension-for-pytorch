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

#include "gemm.h"
#include <torch/all.h>
#include <ATen/native/Resize.h>
#include <cstdlib>

namespace kpex {
// Actual implementation for mm,mm.out,addmm,addmm.out variations (matrix multiply and addition). Returns alpha*m1@m2+beta*self to result tensor
static void addmm_out_impl_(at::Tensor &result, const at::Tensor &self, const at::Tensor &m1, const at::Tensor &m2, const at::Scalar& beta, const at::Scalar& alpha) {
    // Following addmm_impl_cpu_() at aten/src/ATen/native/LinearAlgebra.cpp
    RECORD_FUNCTION("KPEX::addmm_out_impl_", std::vector<c10::IValue>{self, m1, m2});

    TORCH_INTERNAL_ASSERT(self.dim() == 2 && m1.dim() == 2 && m2.dim() == 2);

    TORCH_CHECK(
        m1.dtype() == m2.dtype(),
        "kpex: expected m1 and m2 to have the same dtype, but got: ", m1.dtype(), " != ", m2.dtype()
    )
    
    // Array access is faster than .size(n) and .stride(n)
    const auto self_sizes = self.sizes();
    auto m1_strides = m1.strides();
    auto m1_sizes = m1.sizes();
    auto m2_strides = m2.strides();
    auto m2_sizes = m2.sizes();

    TORCH_CHECK(
        self_sizes[0] == m1_sizes[0] && self_sizes[1] == m2_sizes[1],
        "kpex: input shape is incompatible with matrix multiplication (",
        m1_sizes[0], "x", m1_sizes[1], " @ ", m2_sizes[0], "x", m2_sizes[1], " != ",
        self_sizes[0], "x", self_sizes[1], ")");

    at::native::resize_output(result, self_sizes); // Resize to proper size in case a user sent an empty output tensor
    const auto result_strides = result.strides();
    const auto result_sizes = result.sizes();
    // resulting matrix has zero elements
    if (result.numel() == 0) {
        return;
    }
    // Inner Dimension = 0
    if (m1_sizes[1] == 0) { // empty tensor multiplication
        if (beta.toComplexDouble() == 0.0) {
            result.zero_(); // no bias, return zeroed output
        } else {
            if (!self.is_same(result)) { // no matrix multiplication, result is just what's in beta*self
                result.copy_(self);
            }
            result.mul_(beta);
        }
        return;
    }
    // Safety check for an in-place operation; current result will be added to previously calculated in self
    if (beta.toComplexDouble() != 0.0 && !self.is_same(result)) {
        result.copy_(self);
    }

    // Setup and call KuDNN GEMM

    // Pytorch handles initialization and update of number of work threads
    int numThreads = at::get_num_threads();

    KuDNN::Layout srcLayout;
    at::Tensor a;
    CHECK_AND_SET_LAYOUT(m1_strides, m1_sizes, m1, a, srcLayout);
    KuDNN::Layout weiLayout;
    at::Tensor b;
    CHECK_AND_SET_LAYOUT(m2_strides, m2_sizes, m2, b, weiLayout);
    KuDNN::Layout dstLayout;
    at::Tensor c;
    CHECK_AND_SET_LAYOUT(result_strides, result_sizes, result, c, dstLayout);

    auto a_sizes = a.sizes();
    auto b_sizes = b.sizes();
    int M = a_sizes[0];
    int K = a_sizes[1];
    int N = b_sizes[1];

    KuDNN::Shape srcShape(M, K);
    KuDNN::Shape weiShape(K, N);
    KuDNN::Shape dstShape(M, N);
    // Get actual strides from PyTorch tensors
    auto a_strides = a.strides();
    auto b_strides = b.strides();
    auto c_strides = c.strides();
    // Pass strides to TensorInfo constructor!
    KuDNN::Shape srcStrides(a_strides[0], a_strides[1]);
    KuDNN::Shape weiStrides(b_strides[0], b_strides[1]);
    KuDNN::Shape dstStrides(c_strides[0], c_strides[1]);
    // Initialization of TensorInfo. To be modified in DISPATCH_TO_CPP_TYPES macro scope
    KuDNN::TensorInfo srcTensor{srcShape, KuDNN::Element::MatchType<__bf16>(), srcLayout, srcStrides};
    KuDNN::TensorInfo weiTensor{weiShape, KuDNN::Element::MatchType<__bf16>(), weiLayout, weiStrides};
    KuDNN::TensorInfo dstTensor{dstShape, KuDNN::Element::MatchType<__bf16>(), dstLayout, dstStrides};

    DISPATCH_TO_CPP_TYPES(a.scalar_type(), "addmm_out_impl_", {
        // Inside this lambda, 'cpp_type' is the C++ type corresponding to self.scalar_type()
        using a_cpp_type = cpp_type;
        srcTensor = {srcShape, KuDNN::Element::MatchType<a_cpp_type>(), srcLayout, srcStrides};
    });
    DISPATCH_TO_CPP_TYPES(b.scalar_type(), "addmm_out_impl_", {
        using b_cpp_type = cpp_type;
        weiTensor = {weiShape, KuDNN::Element::MatchType<b_cpp_type>(), weiLayout, weiStrides};
    });
    DISPATCH_TO_CPP_TYPES(c.scalar_type(), "addmm_out_impl_", {
        using c_cpp_type = cpp_type;
        dstTensor = {dstShape, KuDNN::Element::MatchType<c_cpp_type>(), dstLayout, dstStrides};
    });

    // torch.addmm: out = alpha*(mat1 @ mat2) + beta*input // this is not in-place version  (out != self)
    // GEMM:        out = alpha*(mat1 @ mat2) + beta*out (+ bias: optional)
    // const void *src (m1) input activations [M,K]
    // const void *wei (m2) weights           [K,N]
    // const void *dst (result)               [M,N]

    float alpha_f = alpha.to<float>();
    float beta_f = beta.to<float>();
    KuDNN::Gemm gemmLayer(srcTensor, weiTensor, dstTensor, numThreads); // KuDNN supports pure mm if beta=0
    gemmLayer.Run(a.data_ptr(), b.data_ptr(), c.data_ptr(), alpha_f, beta_f, numThreads);

    // Check if c and result are the same object, before returning result
    if (!c.is_same(result)) {
        result.copy_(c);
    }
}
// Wrapper for addmm.out variation (matrix multiply and addition) implementation. Returns alpha*mat1@mat2+beta*self to result tensor
at::Tensor& addmm_out_impl(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta, const at::Scalar& alpha, at::Tensor &result) {
    RECORD_FUNCTION("KPEX::addmm_out_impl", std::vector<c10::IValue>{});
    auto b_self = expand_size(self, {mat1.sizes()[0], mat2.sizes()[1]}, "addmm_out_impl"); // expand if broadcastable
    {
    at::NoNamesGuard guard; // Disable Named Tensor checks (metadata) inside scope; treat tensors as standard positional arrays
    addmm_out_impl_(result, *b_self, mat1, mat2, beta, alpha);
    }
    return result;
}
// Wrapper for addmm (matrix multiply and addition) implementation. Allocates result tensor and returns alpha*mat1@mat2+beta*self
at::Tensor addmm_impl(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta, const at::Scalar& alpha) {
    RECORD_FUNCTION("KPEX::addmm_impl", std::vector<c10::IValue>{self, mat1, mat2});

    auto output_shape = {mat1.size(0), mat2.size(1)};
    at::Tensor output = at::empty(output_shape, mat1.options());
    output = addmm_out_impl(self, mat1, mat2, beta, alpha, output);
    return output;
}
// Wrapper for mm.out variation (matrix multiply) implementation. Returns self@mat2 to result tensor
at::Tensor& mm_out_impl(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & result) {
    RECORD_FUNCTION("KPEX::mm_out_impl", std::vector<c10::IValue>{});
    {
    at::NoNamesGuard guard; // Disable Named Tensor checks (metadata) inside scope; treat tensors as standard positional arrays
    at::native::resize_output(result, {self.sizes()[0], mat2.sizes()[1]});
    addmm_out_impl_(result, result, self, mat2, 0, 1);
    }
    return result;
}
// Wrapper for mm (matrix multiply) implementation. Allocates result tensor and returns self@mat2
at::Tensor mm_impl(const at::Tensor& self, const at::Tensor& mat2) {
    RECORD_FUNCTION("KPEX::mm_impl", std::vector<c10::IValue>{self, mat2});

    // Create an empty output tensor with the correct shape and options (device, dtype)
    // The resulting shape is [M, N] if inputs are [M, K] and [K, N]
    auto output_shape = {self.size(0), mat2.size(1)};
    // Allocate new memory using options derived from the input tensors
    at::Tensor output = at::empty(output_shape, self.options());
    // Call the out-variant to perform the actual computation efficiently
    output = mm_out_impl(self, mat2, output);
    return output;
}
}   // namespace kpex


namespace {
// Note: This section of "custom_XXX" is for testing and debugging purposes and will be removed in a release version of the package
// Graph Optimization/Fusion for custom operators
// Register in a custom namespace
TORCH_LIBRARY_FRAGMENT(kpex, m) {
    m.def("custom_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor");
    m.def("custom_addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)");
    m.def("custom_mm(Tensor self, Tensor mat2) -> Tensor");
    m.def("custom_mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)");
}
TORCH_LIBRARY_IMPL(kpex, CPU, m) {
    m.impl( TORCH_SELECTIVE_NAME("kpex::custom_addmm"), TORCH_FN((&kpex::addmm_impl)) );
    m.impl( TORCH_SELECTIVE_NAME("kpex::custom_addmm.out"), TORCH_FN((&kpex::addmm_out_impl)) );
    m.impl( TORCH_SELECTIVE_NAME("kpex::custom_mm"), TORCH_FN((&kpex::mm_impl)) );
    m.impl( TORCH_SELECTIVE_NAME("kpex::custom_mm.out"), TORCH_FN((&kpex::mm_out_impl)) );
}

// Operator Registration (Eager/JIT Mode Override) under the standard ATEN namespace
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl( TORCH_SELECTIVE_NAME("aten::addmm"), TORCH_FN((&kpex::addmm_impl)) );
// Convention replaces the period with an underscore in the C++ function name: addmm.out -> addmm_out
  m.impl( TORCH_SELECTIVE_NAME("aten::addmm.out"), TORCH_FN((&kpex::addmm_out_impl)) );
  m.impl( TORCH_SELECTIVE_NAME("aten::mm"), TORCH_FN((&kpex::mm_impl)) );
  m.impl( TORCH_SELECTIVE_NAME("aten::mm.out"), TORCH_FN((&kpex::mm_out_impl)) );
}

} // namespace