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

#ifndef KPEX_TPP_ALPHAFOLD_BIND_H
#define KPEX_TPP_ALPHAFOLD_BIND_H

#include <torch/extension.h>

#include "kupl.h"
#include "gating_attention.h"
#include "outer_product_mean.h"
#include "transition.h"
#include "invariant_point.h"
#include "rigid.h"
#include "global_attention.h"
#include "triangle_multiplication.h"
#include "outer_product_mean.h"
#include "utils/transpose.h"
#include "utils/all_gather.h"
#include "utils/layernorm.h"
#include "utils/linear.h"

namespace alphafold {
inline void bind(pybind11::module &m)
{
    auto submodule = m.def_submodule("alphafold");
    py::class_<GatingAttentionWeight>(submodule, "GatingAttentionWeight")
        .def(py::init<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
                      at::Tensor &>(),
             py::arg("query_w"), py::arg("key_w"), py::arg("value_w"), py::arg("gate_w"), py::arg("gate_b"),
             py::arg("output_w"), py::arg("output_b"));
    submodule.def("gating_attention", &gating_attention, py::arg("q_data"), py::arg("m_data"), py::arg("bias"),
                  py::arg("nonbatched_bias"), py::arg("weights"), py::arg("block_size") = std::nullopt);

    py::class_<OuterProductMeanWeight>(submodule, "OuterProductMeanWeight")
        .def(py::init<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
                      at::Tensor &>(),
             py::arg("input_ln_w"), py::arg("input_ln_b"), py::arg("left_proj_w"), py::arg("left_proj_b"),
             py::arg("right_proj_w"), py::arg("right_proj_b"), py::arg("output_w"), py::arg("output_b"));
    submodule.def("outer_product_mean", &outer_product_mean, py::arg("act"), py::arg("mask"), py::arg("weights"),
                  py::arg("left_block_size") = std::nullopt, py::arg("right_block_size") = std::nullopt,
                  py::arg("no_mpi") = false);

    py::class_<GlobalAttentionWeight>(submodule, "GlobalAttentionWeight")
        .def(py::init<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
                      at::Tensor &>(),
             py::arg("query_w"), py::arg("key_w"), py::arg("value_w"), py::arg("gate_w"), py::arg("gate_b"),
             py::arg("output_w"), py::arg("output_b"));
    submodule.def("global_attention", &global_attention, py::arg("q_data"), py::arg("m_data"), py::arg("q_mask"),
                  py::arg("weights"));

    py::class_<TransitionWeight>(submodule, "TransitionWeight")
        .def(py::init<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &>(),
             py::arg("input_ln_w"), py::arg("input_ln_b"), py::arg("linear1_w"), py::arg("linear1_b"),
             py::arg("linear2_w"), py::arg("linear2_b"));
    submodule.def("transition", &transition, py::arg("act"), py::arg("weights"));

    submodule.def("rigid_rot_vec_mul", &rigid_rot_vec_mul, py::arg("pts"), py::arg("rot_mats"),
                  py::arg("trans") = std::nullopt);
    submodule.def("rigid_rot_matmul", &rigid_rot_matmul, py::arg("a"), py::arg("b"));

    py::class_<InvariantPointAttentionWeight>(submodule, "InvariantPointAttentionWeight")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, at::Tensor &, at::Tensor &, at::Tensor &,
                      at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
                      at::Tensor &, at::Tensor &, at::Tensor &>(),
             py::arg("c_s"), py::arg("c_z"), py::arg("c_hidden"), py::arg("no_heads"), py::arg("no_qk_points"),
             py::arg("no_v_points"), py::arg("linear_q_w"), py::arg("linear_q_b"), py::arg("linear_kv_w"),
             py::arg("linear_kv_b"), py::arg("linear_q_points_w"), py::arg("linear_q_points_b"),
             py::arg("linear_kv_points_w"), py::arg("linear_kv_points_b"), py::arg("linear_b_w"), py::arg("linear_b_b"),
             py::arg("head_weights"), py::arg("linear_out_w"), py::arg("linear_out_b"));
    submodule.def("invariant_point_attention", &invariant_point_attention, py::arg("s"), py::arg("z"),
                  py::arg("rigid_trans"), py::arg("rigid_rot_mats"), py::arg("mask"), py::arg("weights"));

    submodule.def("rigid_rot_vec_mul", &rigid_rot_vec_mul, py::arg("pts"), py::arg("rot_mats"),
                  py::arg("trans") = std::nullopt);
    submodule.def("rigid_rot_matmul", &rigid_rot_matmul, py::arg("a"), py::arg("b"));

    py::class_<TriangleMultiplicationWeight>(submodule, "TriangleMultiplicationWeight")
        .def(py::init<bool, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
                      at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
                      at::Tensor &, at::Tensor &, at::Tensor &>(),
             py::arg("is_incoming"), py::arg("input_ln_w"), py::arg("input_ln_b"), py::arg("left_proj_w"),
             py::arg("left_proj_b"), py::arg("right_proj_w"), py::arg("right_proj_b"), py::arg("left_gate_w"),
             py::arg("left_gate_b"), py::arg("right_gate_w"), py::arg("right_gate_b"), py::arg("gating_w"),
             py::arg("gating_b"), py::arg("center_ln_w"), py::arg("center_ln_b"), py::arg("output_proj_w"),
             py::arg("output_proj_b"));
    submodule.def("triangle_multiplication", &triangle_multiplication, py::arg("act"), py::arg("mask"),
                  py::arg("weights"));

    auto mpimodule = m.def_submodule("mpi");
    mpimodule.def("comm_init", &comm_init, py::arg("world_size"), py::arg("rank"), py::arg("buffer_size"));
    mpimodule.def("comm_fini", &comm_fini);
    mpimodule.def("all_gather", &af2_all_gather, py::arg("data"), py::arg("m"), py::arg("n"));
    mpimodule.def("all2all", &af2_transpose, py::arg("data"), py::arg("m"), py::arg("n"));

    auto opsmodule = m.def_submodule("ops");
    opsmodule.def("layernorm", &layernorm, py::arg("act"), py::arg("weight_"), py::arg("bias_"));
    opsmodule.def("linear", &linear, py::arg("act"), py::arg("weight"), py::arg("bias") = std::nullopt,
                  py::arg("result_") = std::nullopt);
}
} // namespace alphafold
#endif
