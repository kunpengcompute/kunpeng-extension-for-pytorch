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

#pragma once

#include <ATen/ATen.h>
#include "utils.h"

namespace kpex {
at::Tensor linear_act_impl(const at::Tensor& self, const at::Tensor& weight, const std::optional<at::Tensor>& bias_opt);
at::Tensor& linear_act_out_impl(const at::Tensor& self, const at::Tensor& weight, const std::optional<at::Tensor>& bias_opt, at::Tensor &result);
}   // namespace kpex
