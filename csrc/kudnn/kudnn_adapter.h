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

#ifndef KPEX_KUDNN_ADAPTER_H
#define KPEX_KUDNN_ADAPTER_H

#include "kudnn.hpp"
#include <ATen/ATen.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <c10/core/ScalarType.h>
#include <ATen/NativeFunctions.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <torch/extension.h>
#include <arm_neon.h>
#include <vector>
#include <unordered_map>
#include <optional>
#include "utils/check.h"

namespace py = pybind11;

namespace kudnn {
KuDNN::TensorInfo getKuDNNTensor(const at::Tensor& tensor);

bool isValidateTensor(const at::Tensor& input);

at::Tensor kudnn_linear(const at::Tensor& input, const at::Tensor& weight, const std::optional<at::Tensor>& bias);

at::Tensor kudnn_conv2d(
    const at::Tensor& input, const at::Tensor& weight, const std::optional<at::Tensor>& bias_opt,
    py::object stride, py::object padding, py::object dilation, int64_t groups); // temporary remove C10::SymInt groups

at::Tensor kudnn_conv3d(
    const at::Tensor& input, const at::Tensor& weight, const std::optional<at::Tensor>& bias_opt,
    py::object stride, py::object padding, py::object dilation, int64_t groups);
}
#endif