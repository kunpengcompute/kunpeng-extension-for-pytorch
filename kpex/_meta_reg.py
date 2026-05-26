#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# KPEX is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#        http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#

import torch
# Register meta functions for the dynamo compiler run
# Purpose is to notify the compiler the output size without actual computation
# Each meta function corresponds to a custom implementation register_fake("kpex::custom_implementation").
# It is performing sanity checks on function inputs and calculates the proper output shape.

@torch.library.register_fake("kpex::custom_mm")
@torch.library.register_fake("kpex::custom_mm.out")
def meta_custom_mm(self, mat2, out=None):
    torch._check(self.dim() == 2, lambda: "self must be 2D")
    torch._check(mat2.dim() == 2, lambda: "mat2 must be 2D")
    torch._check(self.dtype == mat2.dtype, lambda: f"Expected self and mat2 to have the same dtype, but got: {self.dtype} != {mat2.dtype}")
    N, M1 = self.shape
    M2, P = mat2.shape
    torch._check(M1 == M2, lambda: f"self and mat2 must have same reduction dim, but got [{N}, {M1}] X [{M2}, {P}].",)
    res_shape = (N, P)
    if out is not None:
        torch._check(out.dtype == self.dtype, 
                    lambda: f"Expected out dtype {self.dtype}, but got {out.dtype}")
        torch._check(out.device == self.device, 
                    lambda: f"Expected out device {self.device}, but got {out.device}")
        if out.shape != torch.Size(res_shape):
            out.resize_(res_shape)
        return out
    return torch.empty(res_shape, device=self.device, dtype=self.dtype)

@torch.library.register_fake("kpex::custom_addmm")
@torch.library.register_fake("kpex::custom_addmm.out")
def meta_custom_addmm(self, mat1, mat2, beta=1, alpha=1, out=None):
    torch._check(mat1.dim() == 2, lambda: "mat1 must be 2D")
    torch._check(mat2.dim() == 2, lambda: "mat2 must be 2D")
    torch._check(mat1.dtype == mat2.dtype, lambda: f"Expected mat1 and mat2 to have the same dtype, but got: {mat1.dtype} != {mat2.dtype}")
    N, M1 = mat1.shape
    M2, P = mat2.shape
    torch._check(M1 == M2, lambda: f"mat1 and mat2 must have same reduction dim, but got [{N}, {M1}] X [{M2}, {P}].",)
    res_shape = (N, P)
    try:
        final_shape = torch.broadcast_shapes(self.shape, res_shape)
    except RuntimeError:
        torch._check(False, lambda: f"self {self.shape} is not broadcastable to {res_shape}")
    if out is not None:
        torch._check(out.dtype == mat1.dtype, 
                    lambda: f"Expected out dtype {mat1.dtype}, but got {out.dtype}")
        torch._check(out.device == mat1.device, 
                    lambda: f"Expected out device {mat1.device}, but got {out.device}")
        if out.shape != torch.Size(res_shape):
            out.resize_(res_shape)
        return out
    return torch.empty(res_shape, device=self.device, dtype=self.dtype)

@torch.library.register_fake("kpex::linear_act")
@torch.library.register_fake("kpex::linear_act.out")
def meta_linear_act(self, weight, bias=None, out=None):
    torch._check(self.dim() >= 1, lambda: f"self must be at least 1D, got {self.dim()}D")
    torch._check(weight.dim() == 2, lambda: f"weight must be 2D, got {weight.dim()}D")
    torch._check(self.dtype == weight.dtype, lambda: f"Expected self and weight to have the same dtype, but got: {self.dtype} != {weight.dtype}")
    M1 = self.shape[-1]
    P, M2 = weight.shape # [out_features, in_features]
    torch._check(M1 == M2, lambda: f"self and weight must have same reduction dim, but got [batch dimensions, {M1}] X [{M2}, {P}].",)
    if self.dim() == 1:
        res_shape = [P]
    else:
        res_shape = list(self.shape)
        res_shape[-1] = P
    if bias is not None:
        try:
            final_shape = torch.broadcast_shapes(bias.shape, tuple(res_shape))
            # Ensure bias doesn't introduce extra dimensions
            torch._check(final_shape == tuple(res_shape), 
                        lambda: f"Bias {bias.shape} is too large; would expand output to {final_shape}")
        except RuntimeError:
            torch._check(False, lambda: f"Bias {bias.shape} is not broadcastable to {res_shape}")
    if out is not None:
        torch._check(out.dtype == self.dtype, 
                    lambda: f"Expected out dtype {self.dtype}, but got {out.dtype}")
        torch._check(out.device == self.device, 
                    lambda: f"Expected out device {self.device}, but got {out.device}")
        if out.shape != torch.Size(res_shape):
            out.resize_(res_shape)
        return out
    return torch.empty(res_shape, device=self.device, dtype=self.dtype)
