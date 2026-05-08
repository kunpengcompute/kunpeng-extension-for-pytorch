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

#include <ATen/native/cpu/utils.h>
#include <ATen/ops/empty.h>
#include <ATen/record_function.h>
#include "kutacc.h"

#include "rigid.h"
#include "utils/TensorWrapper.h"

namespace alphafold {
at::Tensor rigid_rot_vec_mul(at::Tensor &pts, at::Tensor &rot_mats, std::optional<at::Tensor> trans)
{
    KPEX_CHECK(pts.dim() >= 1, "parameter pts's dim is less than 1, it will cause out of bound access");
    int64_t dim = pts.dim() - 1;
    KPEX_CHECK(pts.strides()[dim] == 1, pts.strides());
    KPEX_CHECK(rot_mats.dim() == dim + 2, rot_mats.dim(), dim);
    KPEX_CHECK(rot_mats.strides()[dim] == 3 && rot_mats.strides()[dim + 1] == 1, rot_mats.strides());
    kutacc_tensor_h trans_ = nullptr;
    if (trans.has_value()) {
        KPEX_CHECK(trans->dim() == dim + 1, trans->dim(), dim);
        KPEX_CHECK(trans->strides()[dim] == 1, trans->strides());
        auto trans_tensor = trans.value();

        auto trans_tw = convert_to_tensor_wrapper(trans_tensor);
        trans_ = convert_to_tensor_wrapper(trans.value()).get_tensor();
        auto out = at::empty(pts.sizes(), pts.options());

        auto pts_tw = convert_to_tensor_wrapper(pts);
        auto rot_mats_tw = convert_to_tensor_wrapper(rot_mats);
        auto out_tw = convert_to_tensor_wrapper(out);
        kutacc_af2_rigid_rot_vec_mul(pts_tw.get_tensor(), rot_mats_tw.get_tensor(), out_tw.get_tensor(),
                                     trans_tw.get_tensor());
        return out;
    } else {
        auto out = at::empty(pts.sizes(), pts.options());

        auto pts_tw = convert_to_tensor_wrapper(pts);
        auto rot_mats_tw = convert_to_tensor_wrapper(rot_mats);
        auto out_tw = convert_to_tensor_wrapper(out);
        kutacc_af2_rigid_rot_vec_mul(pts_tw.get_tensor(), rot_mats_tw.get_tensor(), out_tw.get_tensor(), nullptr);
        return out;
    }

    auto out = at::empty(pts.sizes(), pts.options());

    auto pts_tw = convert_to_tensor_wrapper(pts);
    auto rot_mats_tw = convert_to_tensor_wrapper(rot_mats);
    auto out_tw = convert_to_tensor_wrapper(out);
    kutacc_af2_rigid_rot_vec_mul(pts_tw.get_tensor(), rot_mats_tw.get_tensor(), out_tw.get_tensor(), trans_);
    return out;
}

at::Tensor rigid_rot_matmul(at::Tensor &a, at::Tensor &b)
{
    KPEX_CHECK(a.dim() >= 2, "parameter a's dim is less than 2, it will cause out of bound access");
    int64_t dim = a.dim() - 2;
    KPEX_CHECK(b.dim() == dim + 2, b.dim(), dim);
    KPEX_CHECK(a.strides()[dim] == 3 && a.strides()[dim + 1] == 1, a.strides());
    KPEX_CHECK(b.strides()[dim] == 3 && b.strides()[dim + 1] == 1, b.strides());
    KPEX_CHECK(a.scalar_type() == c10::kFloat, a.scalar_type());
    KPEX_CHECK(b.scalar_type() == c10::kFloat, b.scalar_type());

    auto out = at::empty(b.sizes(), b.options());

    auto a_tw = convert_to_tensor_wrapper(a);
    auto b_tw = convert_to_tensor_wrapper(b);
    auto out_tw = convert_to_tensor_wrapper(out);
    kutacc_af2_rigid_rot_matmul(a_tw.get_tensor(), b_tw.get_tensor(), out_tw.get_tensor());
    return out;
}

} // namespace alphafold
