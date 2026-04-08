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

import copy
import time
import types

import torch
from torch import nn
import numpy as np
import torch.distributed as dist
import kpex._C as kernel
import kpex
import os

bf16 = os.environ.get("AF2_BF16") == "1"
distributed_mpi = os.environ.get("DISTRIBUTED_MPI") == "1"
distributed_extrabefore_outer = os.environ.get("DISTRIBUTED_EXTRABEFORE_OUTER") == "1"
distributed_extraafter_outer = os.environ.get("DISTRIBUTED_EXTRAAFTER_OUTER") == "1"
distributed_noextrabefore_outer = (
    os.environ.get("DISTRIBUTED_NOEXTRABEFORE_OUTER") == "1"
)
distributed_gatingattention = os.environ.get("DISTRIBUTED_GATINGATTENTION") == "1"
distributed_noextraafter_outer = os.environ.get("DISTRIBUTED_NOEXTRAAFTER_OUTER") == "1"
distributed_embedding_triangle = os.environ.get("DISTRIBUTED_EMBEDDING_TRIANGLE") == "1"
save_active_values = os.environ.get("SAVE_ACTIVE_VALUES") == "1"

class ParameterError(Exception):
    def __init__(self, error_msg):
        print(f"Parameter Error: {error_msg}")
        super().__init__(error_msg)

def gating_attention_forward(self, q_data, m_data, bias, nonbatched_bias=torch.Tensor(), block_size=None):

    if not hasattr(self, "kpex_weights"):
        self.kpex_weights = kernel.alphafold.GatingAttentionWeight(
            self.query_w.permute(1, 2, 0),
            self.key_w.permute(1, 2, 0),
            self.value_w.permute(1, 2, 0),
            self.gating_w.permute(1, 2, 0),
            self.gating_b,
            self.output_w.permute(2, 0, 1),
            self.output_b,
        )
    act = q_data.to(torch.bfloat16)
    out = kernel.alphafold.gating_attention(
        act,
        act,
        bias.to(torch.bfloat16),
        nonbatched_bias.to(torch.bfloat16),
        self.kpex_weights,
        block_size,
    )
    return out

def global_attention_forward(self, q_data, m_data, q_mask, bias):
    if not hasattr(self, "kpex_weights"):
        self.kpex_weights = kernel.alphafold.GlobalAttentionWeight(
            self.query_w.permute(1, 2, 0),
            self.key_w.permute(1, 0),
            self.value_w.permute(1, 0),
            self.gating_w.permute(1, 2, 0),
            self.gating_b,
            self.output_w.permute(2, 0, 1),
            self.output_b
        )
    act = q_data.to(torch.bfloat16)
    out = kernel.alphafold.global_attention(
        act,
        act,
        q_mask.to(torch.bfloat16),
        self.kpex_weights,
    )
    return out

def outer_product_mean_forward(
    self, act, mask, left_block_size=None, right_block_size=None, no_mpi=False
):
    if not hasattr(self, "kpex_weights"):
        self.kpex_weights = kernel.alphafold.OuterProductMeanWeight(
            self.layer_norm_input.weight,
            self.layer_norm_input.bias,
            self.left_projection.weight,
            self.left_projection.bias,
            self.right_projection.weight,
            self.right_projection.bias,
            self.output_w.permute(2, 0, 1),
            self.output_b,
        )
    out = kernel.alphafold.outer_product_mean(
        act.to(torch.bfloat16),
        mask.to(torch.bfloat16),
        self.kpex_weights,
        left_block_size,
        right_block_size,
        no_mpi,
    )
    return out

def transition_forward(self, act, mask):
    if not hasattr(self, "kpex_weights"):
        self.kpex_weights = kernel.alphafold.TransitionWeight(
            self.input_layer_norm.weight,
            self.input_layer_norm.bias,
            self.transition1.weight,
            self.transition1.bias,
            self.transition2.weight,
            self.transition2.bias
        )
    out = kernel.alphafold.transition(act.to(torch.bfloat16), self.kpex_weights)
    return out

def invariant_point_forward(self, s, z, r, mask):
    if not hasattr(self, "kpex_weights"):
        self.kpex_weights = kernel.alphafold.InvariantPointAttentionWeight(
            self.c_s,
            self.c_z,
            self.c_hidden,
            self.no_heads,
            self.no_qk_points,
            self.no_v_points,
            self.linear_q.weight,
            self.linear_q.bias,
            self.linear_kv.weight,
            self.linear_kv.bias,
            self.linear_q_points.weight,
            self.linear_q_points.bias,
            self.linear_kv_points.weight,
            self.linear_kv_points.bias,
            self.linear_b.weight,
            self.linear_b.bias,
            self.head_weights,
            self.linear_out.weight,
            self.linear_out.bias,
        )
    out = kernel.alphafold.invariant_point_attention(
        s.to(torch.bfloat16),
        z.to(torch.bfloat16),
        r._trans,
        r._rots.get_rot_mats(),
        mask.to(torch.bfloat16),
        self.kpex_weights,
    )
    return out
        
def triangleMultiplication_forward(self, act, mask):
    if not hasattr(self, "kpex_weights"):
        self.kpex_weights = kernel.alphafold.TriangleMultiplicationWeight(
            self.c_equation == "kjc,kic->ijc",
            self.layer_norm_input.weight,
            self.layer_norm_input.bias,
            self.left_projection.weight,
            self.left_projection.bias,
            self.right_projection.weight,
            self.right_projection.bias,
            self.left_gate.weight,
            self.left_gate.bias,
            self.right_gate.weight,
            self.right_gate.bias,
            self.gating_linear.weight,
            self.gating_linear.bias,
            self.center_layer_norm.weight,
            self.center_layer_norm.bias,
            self.output_projection.weight,
            self.output_projection.bias,
        )
    out = kernel.alphafold.triangle_multiplication(
        act.to(torch.bfloat16),
        mask.to(torch.bfloat16),
        self.kpex_weights,
    )
    return out

def rot_vec_mul(r, t):
    return kernel.alphafold.rigid_rot_vec_mul(t, r)

def rot_to_quat(
    rot: torch.Tensor
):
    if (rot.shape[-2:] != (3,3)):
        return ValueError("Input rotation is incorrectly shaped")

    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

    k = [
        [xx + yy + zz, zy - yz, xz - zx, yx - xy],
        [zy - yz, xx - yy - zz, xy + yx, xz + zx],
        [xz - zx, xy + yx, yy - xx - zz, yz + zy],
        [yx - xy, xz + zx, yz + zy, zz - xx - yy],
    ]

    k = (1. / 3.) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)
    _, vectors = np.linalg.eigh(k.numpy())
    return torch.from_numpy(vectors[..., -1])

def MSARowAttentionWithPairBias_forward(self, msa_act, msa_mask, pair_act):
    if msa_act.dim() != 3:
        raise ParameterError(f"msa_act dim is not 3 dimensions : {msa_act.dim()}")
    if msa_mask.dim() != 2:
        raise ParameterError(f"msa_mask dim is not 2 dimensions : {msa_mask.dim()}")
    bias = (1e9 * (msa_mask - 1.0))[:, None, None, :]
    msa_act = kernel.ops.layernorm(
        msa_act.to(torch.bfloat16), self.query_norm.weight, self.query_norm.bias
    )
    pair_act = kernel.ops.layernorm(
        pair_act.to(torch.bfloat16), self.feat_2d_norm.weight, self.feat_2d_norm.bias
    )
    if not hasattr(self, "feat_2d_weights_t"):
        self.feat_2d_weights_t = (
            self.feat_2d_weights.transpose(0, 1).to(torch.bfloat16).contiguous()
        )
    nonbatched_bias = kernel.ops.linear(self.feat_2d_weights_t, pair_act)
    if distributed_mpi:
        nonbatched_bias = kpex._C.mpi.all_gather(
            nonbatched_bias, nonbatched_bias.size(0), nonbatched_bias.size(2)
        )
    msa_act = self._slice_attention(msa_act, msa_act, bias, nonbatched_bias)
    return msa_act


def MSAColumnAttention_forward(self, msa_act, msa_mask):
    if msa_act.dim() != 3:
        raise ParameterError(f"msa_act dim is not 3 dimensions : {msa_act.dim()}")
    if msa_mask.dim() != 2:
        raise ParameterError(f"msa_mask dim is not 2 dimensions : {msa_mask.dim()}")
    msa_act = torch.swapaxes(msa_act, -2, -3)
    msa_mask = torch.swapaxes(msa_mask, -1, -2)
    bias = (1e9 * (msa_mask - 1.0))[:, None, None, :]
    if bias.dim() != 4:
        raise ParameterError(f"bias dim is not 4 dimensions : {bias.dim()}")
    msa_act = kernel.ops.layernorm(
        msa_act.to(torch.bfloat16), self.query_norm.weight, self.query_norm.bias
    )
    msa_act = self.attention(msa_act, msa_act, bias)
    msa_act = torch.swapaxes(msa_act, -2, -3)
    return msa_act


def TriangleAttention_forward(self, pair_act, pair_mask):
    if pair_act.dim() != 3:
        raise ParameterError(f"msapair_act_act dim is not 3 dimensions : {pair_act.dim()}")
    if pair_mask.dim() != 2:
        raise ParameterError(f"pair_mask dim is not 2 dimensions : {pair_mask.dim()}")
    distributed_triangle = os.environ.get("DISTRIBUTED_TRIANGLE") == "1"
    if self.c_orientation == "per_column":
        if distributed_mpi and distributed_extraafter_outer and distributed_triangle:
            pair_act = kpex._C.mpi.all2all(
                pair_act.transpose(0, 1),
                kpex.tpp.alphafold.pair_act_n_seq,
                kpex.tpp.alphafold.pair_act_n_res,
            )
        else:
            pair_act = torch.swapaxes(pair_act, -2, -3)
        pair_mask = torch.swapaxes(pair_mask, -1, -2)
    bias = (1e9 * (pair_mask - 1.0))[:, None, None, :]
    if bias.dim() != 4:
        raise ParameterError(f"bias dim is not 4 dimensions : {bias.dim()}")
    pair_act = kernel.ops.layernorm(
        pair_act.to(torch.bfloat16), self.query_norm.weight, self.query_norm.bias
    )
    if not hasattr(self, "feat_2d_weights_t"):
        self.feat_2d_weights_t = (
            self.feat_2d_weights.transpose(0, 1).to(torch.bfloat16).contiguous()
        )
    nonbatched_bias = kernel.ops.linear(self.feat_2d_weights_t, pair_act)
    if distributed_mpi and distributed_extraafter_outer and distributed_triangle:
        nonbatched_bias = kpex._C.mpi.all_gather(
            nonbatched_bias, nonbatched_bias.size(0), nonbatched_bias.size(2)
        )

    pair_act = self._slice_attention(pair_act, pair_act, bias, nonbatched_bias)
    if self.c_orientation == "per_column":
        if distributed_mpi and distributed_extraafter_outer and distributed_triangle:
            pair_act = kpex._C.mpi.all2all(
                pair_act.transpose(0, 1),
                kpex.tpp.alphafold.pair_act_n_res,
                kpex.tpp.alphafold.pair_act_n_seq,
            )
        else:
            pair_act = torch.swapaxes(pair_act, -2, -3)
    return pair_act


def MSAColumnGlobalAttention_forward(self, msa_act, msa_mask):
    if msa_act.dim() != 3:
        raise ParameterError(f"msa_act dim is not 3 dimensions : {msa_act.dim()}")
    if msa_mask.dim() != 2:
        raise ParameterError(f"msa_mask dim is not 2 dimensions : {msa_mask.dim()}")
    msa_act = torch.swapaxes(msa_act, -2, -3)
    msa_mask = torch.swapaxes(msa_mask, -1, -2)
    bias = (1e9 * (msa_mask - 1.0))[:, None, None, :]
    if bias.dim() != 4:
        raise ParameterError(f"bias dim is not 4 dimensions : {bias.dim()}")
    msa_act = kernel.ops.layernorm(
        msa_act.to(torch.bfloat16), self.query_norm.weight, self.query_norm.bias
    )
    msa_mask = torch.unsqueeze(msa_mask, -1)
    msa_act = self.attention(msa_act, msa_act, msa_mask, bias)

    msa_act = torch.swapaxes(msa_act, -2, -3)
    return msa_act


def ExtraEvoformerIteration_forward(self, msa_act, pair_act, msa_mask, pair_mask):
    rank = 1
    if distributed_mpi:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    if distributed_extrabefore_outer and distributed_mpi:
        if self.extra_iteration == 1:
            msa_act, msa_mask_col, msa_mask_row = row_attention_input_split(
                msa_act, msa_mask, rank, world_size
            )
            pair_act, pair_mask_row, pair_mask_col = triangle_attention_input_split(
                pair_act, pair_mask, rank, world_size
            )
        else:
            pair_mask_row = torch.chunk(pair_mask, world_size, dim=0)[rank]
            pair_mask_col = torch.chunk(pair_mask, world_size, dim=1)[rank]
            msa_mask_row = torch.chunk(msa_mask, world_size, dim=0)[rank]
            msa_mask_col = torch.chunk(msa_mask, world_size, dim=1)[rank]
    else:
        msa_mask_row = msa_mask
        msa_mask_col = msa_mask
        pair_mask_row = pair_mask
        pair_mask_col = pair_mask
    msa_act = msa_act + self.msa_row_attention_with_pair_bias(
        msa_act, msa_mask_row, pair_act=pair_act
    )  # [TODO] CPU usage is low here
    if save_active_values:
        torch.save(msa_act, f"{rank}right_extra_msa_row_attention_after.pt")

    if distributed_mpi:
        msa_act = kpex._C.mpi.all2all(
            msa_act,
            kpex.tpp.alphafold.msa_act_n_seq,
            kpex.tpp.alphafold.msa_act_n_res,
        )
    msa_act = msa_act + self.msa_column_global_attention(msa_act, msa_mask_col)

    if save_active_values:
        torch.save(msa_act, f"{rank}right_extra_column_global_after.pt")

    msa_act = msa_act + self.msa_transition(msa_act, msa_mask_row)
    if save_active_values:
        torch.save(msa_act, f"{rank}right_extra_msa_trans_after.pt")
    pair_act = pair_act + self.outer_product_mean(msa_act, msa_mask)

    if save_active_values:
        torch.save(pair_act, f"{rank}right_extra_outer_after.pt")

    pair_act = pair_act + self.triangle_multiplication_outgoing(pair_act, pair_mask_row)
    pair_act = pair_act + self.triangle_multiplication_incoming(pair_act, pair_mask_row)

    pair_act = pair_act + self.triangle_attention_starting_node(pair_act, pair_mask_row)
    pair_act = pair_act + self.triangle_attention_ending_node(pair_act, pair_mask_col)

    pair_act = pair_act + self.pair_transition(pair_act, pair_mask)

    if (
        distributed_extraafter_outer
        and distributed_mpi
        and self.extra_msa_stack_num_block == self.extra_iteration
    ):
        pair_act = triangle_attention_out_gather(pair_act)

    if distributed_mpi:
        if self.extra_msa_stack_num_block == self.extra_iteration:
            msa_act = msa_transition_out_gather(msa_act)
        else:
            msa_act = kpex._C.mpi.all2all(
                msa_act,
                kpex.tpp.alphafold.msa_act_n_seq,
                kpex.tpp.alphafold.msa_act_n_res,
            )

    if save_active_values:
        torch.save(pair_act, f"{rank}right_extra_pair_trans_after.pt")
    res = {"msa": msa_act}
    del pair_mask
    res["pair"] = pair_act
    del pair_act
    return res


def NoExtraEvoformerIteration_forward(self, msa_act, pair_act, msa_mask, pair_mask):
    if bf16 == True:
        msa_act = msa_act.to(torch.bfloat16)
        pair_act = pair_act.to(torch.bfloat16)
        msa_mask = msa_mask.to(torch.bfloat16)
        pair_mask = pair_mask.to(torch.bfloat16)

    rank = 1
    if distributed_mpi:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    if distributed_noextrabefore_outer and distributed_mpi:
        if self.noextra_iteration == 1:
            msa_act, msa_mask_col, msa_mask_row = row_attention_input_split(
                msa_act, msa_mask, rank, world_size
            )
            pair_act, pair_mask_row, pair_mask_col = triangle_attention_input_split(
                pair_act, pair_mask, rank, world_size
            )
        else:
            pair_mask_row = torch.chunk(pair_mask, world_size, dim=0)[rank]
            pair_mask_col = torch.chunk(pair_mask, world_size, dim=1)[rank]
            msa_mask_row = torch.chunk(msa_mask, world_size, dim=0)[rank]
            msa_mask_col = torch.chunk(msa_mask, world_size, dim=1)[rank]
    else:
        msa_mask_row = msa_mask
        msa_mask_col = msa_mask
        pair_mask_row = pair_mask
        pair_mask_col = pair_mask

    msa_act = msa_act + self.msa_row_attention_with_pair_bias(
        msa_act, msa_mask_row, pair_act=pair_act
    )
    if save_active_values:
        torch.save(msa_act, f"{rank}right_noextra_msa_row_attention_after.pt")
    if distributed_mpi:
        msa_act = kpex._C.mpi.all2all(
            msa_act,
            kpex.tpp.alphafold.msa_act_n_seq,
            kpex.tpp.alphafold.msa_act_n_res,
        )
    msa_act = msa_act + self.msa_column_attention(msa_act, msa_mask_col)
    if save_active_values:
        torch.save(msa_act, f"{rank}right_noextra_column_atten_after.pt")
    msa_act = msa_act + self.msa_transition(msa_act, msa_mask_row)
    if save_active_values:
        torch.save(msa_act, f"{rank}right_noextra_msa_trans_after.pt")
    pair_act = pair_act + self.outer_product_mean(msa_act, msa_mask)
    if save_active_values:
        torch.save(pair_act, f"{rank}right_noextra_outer_after.pt")

    pair_act = pair_act + self.triangle_multiplication_outgoing(pair_act, pair_mask_row)
    pair_act = pair_act + self.triangle_multiplication_incoming(pair_act, pair_mask_row)

    pair_act = pair_act + self.triangle_attention_starting_node(pair_act, pair_mask_row)
    pair_act = pair_act + self.triangle_attention_ending_node(pair_act, pair_mask_col)

    pair_act = pair_act + self.pair_transition(pair_act, pair_mask)

    if (
        distributed_noextraafter_outer
        and distributed_mpi
        and self.evoformer_num_block == self.noextra_iteration
    ):
        pair_act = triangle_attention_out_gather(pair_act)

    if distributed_mpi:
        if self.evoformer_num_block == self.noextra_iteration:
            msa_act = msa_transition_out_gather(msa_act)
        else:
            msa_act = kpex._C.mpi.all2all(
                msa_act,
                kpex.tpp.alphafold.msa_act_n_seq,
                kpex.tpp.alphafold.msa_act_n_res,
            )

    if save_active_values:
        torch.save(pair_act, f"{rank}right_noextra_pair_trans_after.pt")
    res = {"msa": msa_act}
    res["pair"] = pair_act

    return res


def TemplatePairSubStack_forward(self, pair_act, pair_mask):
    if distributed_embedding_triangle and distributed_mpi:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if self.template_pairsubstack_iteration == 1:
            pair_act, pair_mask_row, pair_mask_col = triangle_attention_input_split(
                pair_act, pair_mask, rank, world_size
            )
        else:
            pair_mask_row = torch.chunk(pair_mask, world_size, dim=0)[rank]
            pair_mask_col = torch.chunk(pair_mask, world_size, dim=1)[rank]
    else:
        pair_mask_row = pair_mask
        pair_mask_col = pair_mask

    pair_act = pair_act + self.triangle_attention_starting_node(pair_act, pair_mask_row)
    pair_act_col = pair_act + self.triangle_attention_ending_node(
        pair_act, pair_mask_col
    )
    pair_act = pair_act_col + self.triangle_multiplication_outgoing(
        pair_act_col, pair_mask_row
    )
    pair_act = pair_act + self.triangle_multiplication_incoming(pair_act, pair_mask_row)
    pair_act = pair_act + self.pair_transition(pair_act, pair_mask)

    if (
        distributed_embedding_triangle
        and distributed_mpi
        and self.template_pairsubstack_iteration == self.num_block
    ):
        pair_act = triangle_attention_out_gather(pair_act)

    return pair_act


def msa_act_all2all(msa_act):
    msa_act = msa_act.to(torch.float32)
    world_size = dist.get_world_size()
    msa_act = msa_act.reshape(
        msa_act.size(0), world_size, msa_act.size(1) // world_size, msa_act.size(2)
    )
    msa_act = msa_act.transpose(0, 1).contiguous()
    out_msa_act = torch.zeros_like(msa_act)
    dist.all_to_all_single(out_msa_act, msa_act)
    msa_act = out_msa_act.to(torch.bfloat16)
    msa_act = msa_act.reshape(
        msa_act.size(0) * msa_act.size(1), msa_act.size(2), msa_act.size(3)
    )
    msa_act = msa_act.contiguous()
    return msa_act


def msa_transition_out_gather(msa_act):
    msa_act = kpex._C.mpi.all_gather(
        msa_act, kpex.tpp.alphafold.msa_act_n_seq, kpex.tpp.alphafold.msa_act_n_res
    )
    return msa_act


def row_attention_input_split(msa_act, msa_mask, rank, world_size):
    kpex.tpp.alphafold.msa_act_n_seq = msa_act.size(0)
    kpex.tpp.alphafold.msa_act_n_res = msa_act.size(1)
    msa_act = torch.chunk(msa_act, world_size, dim=0)[rank]
    msa_mask_row = torch.chunk(msa_mask, world_size, dim=0)[rank]
    msa_mask_col = torch.chunk(msa_mask, world_size, dim=1)[rank]
    return msa_act, msa_mask_col, msa_mask_row


def triangle_attention_input_split(pair_act, pair_mask, rank, world_size):
    kpex.tpp.alphafold.pair_act_n_seq = pair_act.size(0)
    kpex.tpp.alphafold.pair_act_n_res = pair_act.size(1)

    pair_act = torch.chunk(pair_act, world_size, dim=0)[rank]
    pair_mask_row = torch.chunk(pair_mask, world_size, dim=0)[rank]
    pair_mask_col = torch.chunk(pair_mask, world_size, dim=1)[rank]

    os.environ["DISTRIBUTED_TRIANGLE"] = "1"
    return pair_act, pair_mask_row, pair_mask_col


def triangle_attention_out_gather(pair_act_col):
    pair_act = kpex._C.mpi.all_gather(
        pair_act_col, kpex.tpp.alphafold.pair_act_n_res, kpex.tpp.alphafold.pair_act_n_seq
    )
    os.environ["DISTRIBUTED_TRIANGLE"] = "0"
    return pair_act


def kpex_alphafold(model, model_config, dtype=torch.float):
    new_model = copy.deepcopy(model)
    evoformer = new_model.model.impl.evoformer
    
    if hasattr(evoformer, "extra_msa_stack"):
        for block in evoformer.extra_msa_stack:
            block.msa_row_attention_with_pair_bias.attention.forward = types.MethodType(
                gating_attention_forward, 
                block.msa_row_attention_with_pair_bias.attention
                )
            block.msa_column_global_attention.attention.forward = types.MethodType(
                global_attention_forward,
                block.msa_column_global_attention.attention
            )
            block.triangle_attention_starting_node.attention.forward = types.MethodType(
                gating_attention_forward, 
                block.triangle_attention_starting_node.attention
                )
            block.triangle_attention_ending_node.attention.forward = types.MethodType(
                gating_attention_forward, 
                block.triangle_attention_ending_node.attention
                )
            block.outer_product_mean.forward = types.MethodType(
                outer_product_mean_forward, 
                block.outer_product_mean
                )
            block.msa_transition.forward = types.MethodType(
                transition_forward, 
                block.msa_transition
                )
            block.pair_transition.forward = types.MethodType(
                transition_forward, 
                block.pair_transition
                )
            
            block.msa_row_attention_with_pair_bias.forward = types.MethodType(
                MSARowAttentionWithPairBias_forward, 
                block.msa_row_attention_with_pair_bias
                )
            block.triangle_attention_starting_node.forward = types.MethodType(
                TriangleAttention_forward, 
                block.triangle_attention_starting_node
                )
            block.triangle_attention_ending_node.forward = types.MethodType(
                TriangleAttention_forward, 
                block.triangle_attention_ending_node
                )
            block.msa_column_global_attention.forward = types.MethodType(
                MSAColumnGlobalAttention_forward, 
                block.msa_column_global_attention
                )
            block.forward = types.MethodType(
                ExtraEvoformerIteration_forward, 
                block
                )
            block.triangle_multiplication_outgoing.forward = types.MethodType(
                triangleMultiplication_forward,
                block.triangle_multiplication_outgoing
            )
            block.triangle_multiplication_incoming.forward = types.MethodType(
                triangleMultiplication_forward,
                block.triangle_multiplication_incoming
            )
    if hasattr(evoformer, "evoformer_iteration"):
        for block in evoformer.evoformer_iteration:
            block.msa_row_attention_with_pair_bias.attention.forward = types.MethodType(
                gating_attention_forward, 
                block.msa_row_attention_with_pair_bias.attention
                )
            block.msa_column_attention.attention.forward = types.MethodType(
                gating_attention_forward, 
                block.msa_column_attention.attention
                )
            block.triangle_attention_starting_node.attention.forward = types.MethodType(
                gating_attention_forward, 
                block.triangle_attention_starting_node.attention
                )
            block.triangle_attention_ending_node.attention.forward = types.MethodType(
                gating_attention_forward, 
                block.triangle_attention_ending_node.attention
                )
            block.outer_product_mean.forward = types.MethodType(
                outer_product_mean_forward, 
                block.outer_product_mean
                )
            block.msa_transition.forward = types.MethodType(
                transition_forward, 
                block.msa_transition
                )
            block.pair_transition.forward = types.MethodType(
                transition_forward, 
                block.pair_transition
                )
            
            block.msa_row_attention_with_pair_bias.forward = types.MethodType(
                MSARowAttentionWithPairBias_forward, 
                block.msa_row_attention_with_pair_bias
                )
            block.triangle_attention_starting_node.forward = types.MethodType(
                TriangleAttention_forward, 
                block.triangle_attention_starting_node
                )
            block.triangle_attention_ending_node.forward = types.MethodType(
                TriangleAttention_forward, 
                block.triangle_attention_ending_node
                )
            block.msa_column_attention.forward = types.MethodType(
                MSAColumnAttention_forward, 
                block.msa_column_attention
                )
            block.forward = types.MethodType(
                NoExtraEvoformerIteration_forward, 
                block
                )
            block.triangle_multiplication_outgoing.forward = types.MethodType(
                triangleMultiplication_forward,
                block.triangle_multiplication_outgoing
                )
            block.triangle_multiplication_incoming.forward = types.MethodType(
                triangleMultiplication_forward,
                block.triangle_multiplication_incoming
                )
    if hasattr(evoformer, "template_embedding"):
        template_pair_sub_stack = evoformer.template_embedding.single_template_embedding.template_pair_stack.template_pair_sub_stack
        for block in template_pair_sub_stack:
            block.triangle_attention_starting_node.attention.forward = types.MethodType(
                gating_attention_forward, 
                block.triangle_attention_starting_node.attention
                )
            block.triangle_attention_ending_node.attention.forward = types.MethodType(
                gating_attention_forward, 
                block.triangle_attention_ending_node.attention
                )
            block.pair_transition.forward = types.MethodType(
                transition_forward, 
                block.pair_transition
                )
            
            block.triangle_attention_starting_node.forward = types.MethodType(
                TriangleAttention_forward, 
                block.triangle_attention_starting_node
                )
            block.triangle_attention_ending_node.forward = types.MethodType(
                TriangleAttention_forward, 
                block.triangle_attention_ending_node
                )
            block.forward = types.MethodType(
                TemplatePairSubStack_forward, 
                block
                )
            block.triangle_multiplication_outgoing.forward = types.MethodType(
                triangleMultiplication_forward,
                block.triangle_multiplication_outgoing
                )
            block.triangle_multiplication_incoming.forward = types.MethodType(
                triangleMultiplication_forward,
                block.triangle_multiplication_incoming
                )

    structure_model = new_model.model.impl.structure_module.model
    structure_model.ipa.forward = types.MethodType(
        invariant_point_forward, 
        structure_model.ipa
        )

    return new_model


