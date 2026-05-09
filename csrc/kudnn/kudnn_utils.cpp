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

namespace kudnn {

static const std::unordered_map<at::ScalarType, KuDNN::Element::TypeT> type_map = {
    {at::kFloat, KuDNN::Element::TypeT::F32}, {at::kBFloat16, KuDNN::Element::TypeT::BF16},
    {at::kHalf, KuDNN::Element::TypeT::F16},  {at::kChar, KuDNN::Element::TypeT::S8},
    {at::kInt, KuDNN::Element::TypeT::S32},
};

static const std::unordered_map<int, KuDNN::Layout> dim_to_layout_map = {
    {1, KuDNN::Layout::A},    {2, KuDNN::Layout::AB},    {3, KuDNN::Layout::ABC},
    {4, KuDNN::Layout::ABCD}, {5, KuDNN::Layout::ABCDE},
};

KuDNN::TensorInfo getKuDNNTensor(const at::Tensor &tensor)
{
    auto sizes = tensor.sizes();
    KuDNN::Shape shape(sizes.data(), sizes.size());
    auto it_type = type_map.find(tensor.scalar_type());
    KuDNN::Element::TypeT type = it_type->second;
    auto it_layout = dim_to_layout_map.find(tensor.dim());
    KuDNN::Layout layout = it_layout->second;
    return KuDNN::TensorInfo(shape, type, layout);
}

bool isKuDNNDTypeUnSupported(const at::Tensor &tensor)
{
    auto dtype = tensor.scalar_type();
    auto it = type_map.find(dtype);
    if (unlikely(it == type_map.end())) {
        std::cout << "dtype not supported" << std::endl;
        return true;
    }
    return false;
}

bool isKuDNNLayoutUnSupported(const at::Tensor &tensor)
{
    // check is dense
    if (unlikely(tensor.layout() != c10::kStrided)) {
        std::cout << "layout not supported" << std::endl;
        return true;
    }

    auto dim = tensor.dim();
    auto it = dim_to_layout_map.find(dim);
    if (unlikely(it == dim_to_layout_map.end())) {
        std::cout << "layout not supported" << std::endl;
        return true;
    }
    return false;
}

bool isKuDNNShapeUnSupported(const at::Tensor &tensor)
{
    if (unlikely(tensor.dim() < 0 || tensor.dim() > 5)) {
        std::cout << "shape not supported" << std::endl;
        return true;
    }
    return false;
}

bool isTensorEmpty(const at::Tensor &tensor)
{
    if (unlikely(tensor.numel() == 0)) {
        std::cout << "tensor is empty" << std::endl;
        return true;
    }
    return false;
}

bool isNestedTensor(const at::Tensor &input)
{
    if (unlikely(input.is_nested())) {
        std::cout << "tensor is nested" << std::endl;
        return true;
    }
    return false;
}

bool isNotContiguousTensor(const at::Tensor &input)
{
    if (unlikely(!input.is_contiguous())) {
        std::cout << "tensor is not contiguous" << std::endl;
        return true;
    }
    return false;
}

bool isValidateTensor(const at::Tensor &input)
{
    if (unlikely(isKuDNNDTypeUnSupported(input) || isKuDNNLayoutUnSupported(input) || isTensorEmpty(input) ||
                 isNestedTensor(input) || isKuDNNShapeUnSupported(input) || isNotContiguousTensor(input))) {
        return false;
    }
    return true;
}

} // namespace kudnn
