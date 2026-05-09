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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <algorithm>

namespace py = pybind11;
using SizeType = KuDNN::SizeType;
using Shape = KuDNN::Shape;

namespace kudnn {
std::vector<int64_t> python_to_int64vector(py::object param, int target_dim)
{
    std::vector<int64_t> int_vec;
    int_vec.reserve(target_dim); // 预分配内存，后续直接填充，避免扩容
    const std::string err_non_neg = "Parameter value must be non-negative";

    if (py::isinstance<py::int_>(param)) {
        const int64_t val = param.cast<int64_t>();
        if (val < 0) {
            throw std::invalid_argument(err_non_neg + ", got " + std::to_string(val));
        }

        for (int i = 0; i < target_dim; ++i) {
            int_vec.push_back(val);
        }
    } else if (py::isinstance<py::tuple>(param) || py::isinstance<py::list>(param)) {
        std::vector<int64_t> temp_vec = param.cast<std::vector<int64_t>>();

        if (temp_vec.size() != static_cast<size_t>(target_dim)) {
            throw std::invalid_argument("Collection length does not match target dimension: expected " +
                                        std::to_string(target_dim) + ", got " + std::to_string(temp_vec.size()));
        }

        for (int i = 0; i < target_dim; ++i) {
            if (temp_vec[i] < 0) {
                throw std::invalid_argument(err_non_neg + ", got " + std::to_string(temp_vec[i]) + " at index " +
                                            std::to_string(i));
            }
        }
        int_vec = std::move(temp_vec);
    } else {
        throw std::invalid_argument("Unsupported param type: only support int, tuple, list");
    }
    return int_vec; // 利用RVO优化，无返回拷贝
}

std::vector<int64_t> python_padding_to_int64vector(py::object param, int target_dim,
                                                   const std::vector<int64_t> &stride_vec,
                                                   const std::vector<int64_t> &dilation_vec,
                                                   const std::vector<int64_t> &kernel_size = {})
{
    std::vector<int64_t> pad_vec;
    pad_vec.reserve(target_dim);

    if (py::isinstance<py::str>(param)) {
        const std::string padding_str = param.cast<std::string>();

        if (padding_str == "valid") {
            // 直接填充0，复用预分配内存
            for (int i = 0; i < target_dim; ++i) {
                pad_vec.push_back(0);
            }
            return pad_vec;
        } else if (padding_str == "same") {
            // 校验stride全为1（直接复用const&传递的stride_vec，无拷贝）
            for (int i = 0; i < target_dim; ++i) {
                if (stride_vec[i] != 1) {
                    throw std::invalid_argument(
                        "padding = 'same' mode does not support any stride values other than 1");
                }
            }

            if (kernel_size.empty() || kernel_size.size() != static_cast<size_t>(target_dim)) {
                throw std::invalid_argument(
                    "kernel_size is required and must match target dimension for 'same' padding");
            }

            for (int i = 0; i < target_dim; ++i) {
                const int64_t dilation_val = dilation_vec[i];
                const int64_t pad_val = (kernel_size[i] - 1) * dilation_val / 2;
                pad_vec.push_back(pad_val);
            }
        } else {
            throw std::invalid_argument("Unsupported padding str: only 'valid' or 'same' are allowed");
        }
    } else {
        // 复用转换逻辑，直接move返回值，避免拷贝
        pad_vec = python_to_int64vector(param, target_dim);
    }

    return pad_vec;
}

std::vector<int64_t> python_to_int64vector_2d(py::object param)
{
    return python_to_int64vector(param, 2);
}

std::vector<int64_t> python_padding_to_int64vector_2d(py::object param, const std::vector<int64_t> &stride_vec,
                                                      const at::Tensor &weight,
                                                      const std::vector<int64_t> &dilation_vec)
{
    const int64_t KH = weight.size(-2), KW = weight.size(-1);
    const std::vector<int64_t> kernel_size = {KH, KW};
    return python_padding_to_int64vector(param, 2, stride_vec, dilation_vec, kernel_size);
}

std::vector<int64_t> python_to_int64vector_3d(py::object param)
{
    return python_to_int64vector(param, 3);
}

std::vector<int64_t> python_padding_to_int64vector_3d(py::object param, const std::vector<int64_t> &stride_vec,
                                                      const at::Tensor &weight,
                                                      const std::vector<int64_t> &dilation_vec)
{
    const int64_t KD = weight.size(2), KH = weight.size(-2), KW = weight.size(-1);
    const std::vector<int64_t> kernel_size = {KD, KH, KW};
    return python_padding_to_int64vector(param, 3, stride_vec, dilation_vec, kernel_size);
}

// 编译期常量，无运行时拷贝
static const std::unordered_map<at::ScalarType, KuDNN::Element::TypeT> SupportedDtypeForConv = {
    {at::kFloat, KuDNN::Element::TypeT::F32},
    {at::kHalf, KuDNN::Element::TypeT::F16},
};

inline bool isSupportedDtypeForConv(const at::Tensor &tensor)
{
    const auto dtype = tensor.scalar_type();
    return SupportedDtypeForConv.find(dtype) != SupportedDtypeForConv.end();
}

static inline std::vector<int64_t> getVec(const std::vector<int64_t> &vec, int64_t offset = 0)
{
    std::vector<int64_t> result;
    result.reserve(vec.size()); // 预分配，与原向量同大小，无扩容

    // 直接遍历const&原向量，计算后填充，无中间拷贝
    for (const auto &val : vec) {
        const int64_t new_val = val + offset;
        if (new_val < 0) {
            throw std::invalid_argument("Converted value must be non-negative, got " + std::to_string(new_val));
        }
        result.push_back(new_val);
    }
    return result; // 转移资源所有权，无拷贝
}

static inline int64_t getOutShapeByPadding(int64_t inSize, int64_t kernelSize, int64_t stride, int64_t dilation_kudnn,
                                           int64_t paddingL, int64_t paddingR)
{
    if (stride <= 0 || kernelSize <= 0 || dilation_kudnn < 0) {
        throw std::invalid_argument("Invalid convolution parameters: stride/kernelSize>0, dilation_kudnn≥0 required.");
    }

    const int64_t effective_kernel_size = kernelSize + (kernelSize - 1) * dilation_kudnn;
    int64_t outSize = (inSize + paddingL + paddingR - effective_kernel_size) / stride + 1;

    return std::max(outSize, static_cast<int64_t>(0));
}

static inline std::vector<int64_t> getOutSize(const at::Tensor &input, const at::Tensor &weight,
                                              const std::vector<int64_t> &strideVec, const Shape &paddingL,
                                              const Shape &paddingR, const std::vector<int64_t> &dilationVec,
                                              int64_t dim)
{
    const int64_t N = input.size(0);
    const int64_t OC = weight.size(0);

    if (dim == 2) {
        const int64_t IH = input.size(-2);
        const int64_t IW = input.size(-1);
        const int64_t KH = weight.size(-2);
        const int64_t KW = weight.size(-1);

        const auto OH = getOutShapeByPadding(IH, KH, strideVec[0], dilationVec[0], static_cast<int64_t>(paddingL[0]),
                                             static_cast<int64_t>(paddingR[0]));
        const auto OW = getOutShapeByPadding(IW, KW, strideVec[1], dilationVec[1], static_cast<int64_t>(paddingL[1]),
                                             static_cast<int64_t>(paddingR[1]));
        return {N, OC, OH, OW}; // 直接初始化，RVO优化，无拷贝
    } else if (dim == 3) {
        const int64_t ID = input.size(2);
        const int64_t IH = input.size(-2);
        const int64_t IW = input.size(-1);
        const int64_t KD = weight.size(2);
        const int64_t KH = weight.size(-2);
        const int64_t KW = weight.size(-1);

        const auto OD = getOutShapeByPadding(ID, KD, strideVec[0], dilationVec[0], static_cast<int64_t>(paddingL[0]),
                                             static_cast<int64_t>(paddingR[0]));
        const auto OH = getOutShapeByPadding(IH, KH, strideVec[1], dilationVec[1], static_cast<int64_t>(paddingL[1]),
                                             static_cast<int64_t>(paddingR[1]));
        const auto OW = getOutShapeByPadding(IW, KW, strideVec[2], dilationVec[2], static_cast<int64_t>(paddingL[2]),
                                             static_cast<int64_t>(paddingR[2]));
        return {N, OC, OD, OH, OW}; // 直接初始化，RVO优化，无拷贝
    } else {
        throw std::invalid_argument("Invalid input dimension. Expected 2 or 3 dimensions, got " + std::to_string(dim));
    }
}

at::Tensor conv_common(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
                       const std::vector<int64_t> &stride_vec, const std::vector<int64_t> &padding_vec,
                       const std::vector<int64_t> &dilation_vec, int64_t groups)
{
    const auto dim = stride_vec.size();

    const at::Tensor &inputContig = input.is_contiguous() ? input : input.contiguous();
    const at::Tensor &weightContig = weight.is_contiguous() ? weight : weight.contiguous();
    const at::Tensor &biasContig = bias.defined() ? (bias.is_contiguous() ? bias : bias.contiguous()) : bias;

    std::vector<int64_t> dilation_vec_kudnn = getVec(dilation_vec, -1); // dilation_kudnn = dilation - 1
    Shape strideShape(stride_vec.data(), stride_vec.size());
    Shape dilationShape(dilation_vec_kudnn.data(), dilation_vec_kudnn.size());

    if (padding_vec.size() != static_cast<size_t>(dim)) {
        throw std::invalid_argument("Invalid padding size. Expected " + std::to_string(dim) + " elements, got " +
                                    std::to_string(padding_vec.size()));
    }

    Shape paddingL(padding_vec.data(), padding_vec.size());
    Shape paddingR(padding_vec.data(), padding_vec.size());

    auto outputSize = getOutSize(input, weight, stride_vec, paddingL, paddingR, dilation_vec_kudnn, dim);

    at::Tensor output = at::empty(outputSize, input.options());

    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    KuDNN::TensorInfo inputTensor = getKuDNNTensor(inputContig);
    KuDNN::TensorInfo weightTensor = getKuDNNTensor(weightContig);
    KuDNN::TensorInfo outputTensor = getKuDNNTensor(output);
    KuDNN::TensorInfo biasTensor = biasContig.defined() ?
                                       getKuDNNTensor(biasContig) :
                                       KuDNN::TensorInfo(KuDNN::Shape(0), outputTensor.GetType(), KuDNN::Layout::A);
    void *biasPtr = biasContig.defined() ? biasContig.data_ptr() : nullptr;

    KuDNN::ConvolutionLayerFWD convFwdLayer(inputTensor, weightTensor, outputTensor, biasTensor, strideShape,
                                            dilationShape, paddingL, paddingR, alg);
    convFwdLayer.Run(inputContig.data_ptr(), weightContig.data_ptr(), output.data_ptr(), biasPtr);

    return output;
}

static std::tuple<at::Tensor, bool> batchify(const at::Tensor &input, const int64_t num_spatial_dims,
                                             const std::string &func_name)
{
    const auto dim_count_no_batch = num_spatial_dims + 1;
    const auto dim_count_batch = dim_count_no_batch + 1;
    const auto is_batched = (input.dim() == dim_count_batch);

    if (!(input.dim() == dim_count_no_batch || is_batched)) {
        throw std::invalid_argument("Expected " + std::to_string(dim_count_no_batch) + "D (unbatched) or " +
                                    std::to_string(dim_count_batch) + "D (batched) input to " + func_name +
                                    ", but got input of size: " + std::to_string(input.dim()));
    }
    // 优化：unsqueeze返回view（非拷贝），std::move转移无拷贝
    return std::make_tuple(is_batched ? input : std::move(input.unsqueeze(0)), is_batched);
}

at::Tensor kudnn_conv2d(const at::Tensor &input, const at::Tensor &weight, const std::optional<at::Tensor> &bias_opt,
                        py::object stride, py::object padding, py::object dilation, int64_t groups)
{
    KPEX_CHECK(isValidateTensor(input), "input is illegal");

    // 转换结果利用RVO，无拷贝；后续传递均为const&，无拷贝
    auto stride_vec = python_to_int64vector_2d(stride);
    auto dilation_vec = python_to_int64vector_2d(dilation);
    auto padding_vec = python_padding_to_int64vector_2d(padding, stride_vec, weight, dilation_vec);

    c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const at::Tensor &bias = *bias_maybe_owned;

    if (bias.defined() && bias.dtype() != input.dtype()) {
        throw std::invalid_argument("Input type (" + std::string(input.dtype().name()) + ") and bias type (" +
                                    std::string(bias.dtype().name()) + ") should be the same");
    }

    auto [input_new, is_batched] = batchify(input, /*num_spatial_dims=*/2, "conv2d");
    auto output = conv_common(input_new, weight, bias, stride_vec, padding_vec, dilation_vec, groups);

    return is_batched ? std::move(output) : std::move(output.squeeze(0));
}

at::Tensor kudnn_conv3d(const at::Tensor &input, const at::Tensor &weight, const std::optional<at::Tensor> &bias_opt,
                        py::object stride, py::object padding, py::object dilation, int64_t groups)
{
    KPEX_CHECK(isValidateTensor(input), "input is illegal");

    auto stride_vec = python_to_int64vector_3d(stride);
    auto dilation_vec = python_to_int64vector_3d(dilation);
    auto padding_vec = python_padding_to_int64vector_3d(padding, stride_vec, weight, dilation_vec);

    c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const at::Tensor &bias = *bias_maybe_owned;

    if (bias.defined() && bias.dtype() != input.dtype()) {
        throw std::invalid_argument("Input type (" + std::string(input.dtype().name()) + ") and bias type (" +
                                    std::string(bias.dtype().name()) + ") should be the same");
    }

    auto [input_new, is_batched] = batchify(input, /*num_spatial_dims=*/3, "conv3d");

    auto output = conv_common(input_new, weight, bias, stride_vec, padding_vec, dilation_vec, groups);

    return is_batched ? std::move(output) : std::move(output.squeeze(0));
}

} // namespace kudnn
