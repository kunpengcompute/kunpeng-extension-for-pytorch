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

#include <c10/core/ScalarType.h>
#include <ATen/ATen.h>
#include "kudnn.hpp"

// F32 (float), F16 (__fp16), BF16 (__bf16), S32 (std::int32_t), S8 (std::int8_t), U8 (std::uint8_t)
// Define the macro to map PyTorch enums to native/LibTorch C++ types
#define DISPATCH_TO_CPP_TYPES(TYPE_ENUM, NAME, ...)                   \
  (void)[&] {                                                         \
    const at::ScalarType _st = TYPE_ENUM;                             \
    switch (_st) {                                                    \
      case at::kFloat:    { using cpp_type = float;         __VA_ARGS__; break; } \
      case at::kHalf:     { using cpp_type = __fp16;        __VA_ARGS__; break; } \
      case at::kBFloat16: { using cpp_type = __bf16;        __VA_ARGS__; break; } \
      case at::kInt:      { using cpp_type = std::int32_t;  __VA_ARGS__; break; } \
      case at::kChar:     { using cpp_type = std::int8_t;   __VA_ARGS__; break; } \
      case at::kByte:     { using cpp_type = std::uint8_t;  __VA_ARGS__; break; } \
      default:                                                                    \
        TORCH_CHECK(false, NAME, " does not support '", at::toString(_st), "'");  \
    }                                                                             \
  }()
// Define the macro check and setup tensor layout
#define CHECK_AND_SET_LAYOUT(tensor_in_strides, tensor_in_sizes, tensor_in, tensor_out, out_layout)             \
  do {                                                                                                          \
    /* Check Physical Row-Major (C-style) or Column-Major (F-style) contiguous */                               \
    if (tensor_in.is_contiguous()) {                                                                            \
        out_layout = KuDNN::Layout::ROW_MAJOR;                                                                   \
        tensor_out = tensor_in; /* new reference to the same object and memory location */                      \
    }                                                                                                           \
    else {                                                                                                      \
        out_layout = KuDNN::Layout::ROW_MAJOR;                                                                   \
        tensor_out = tensor_in.clone(at::MemoryFormat::Contiguous); /* new copy of the tensor in new memory */  \
    }                                                                                                           \
  } while (0)

