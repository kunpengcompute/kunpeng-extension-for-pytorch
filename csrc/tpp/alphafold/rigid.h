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

#include <ATen/core/Tensor.h>
#include <optional>
#include "kutacc.h"
#include "utils/check.h"

namespace alphafold {
    /**
     * @param rot_mats shape [..., 3, 3]
     * @param pts shape [..., 3]
     * @param trans [..., 3]
     * @return shape [..., 3]
     */
    at::Tensor rigid_rot_vec_mul(at::Tensor &pts, at::Tensor &rot_mats, std::optional<at::Tensor> trans);

    /**
     * @param a shape [..., 3, 3]
     * @param b shape [..., 3, 3]
     * @return shape [..., 3, 3]
     */
    at::Tensor rigid_rot_matmul(at::Tensor &a, at::Tensor &b);
}
