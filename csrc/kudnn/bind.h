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

#ifndef KPEX_KUDNN_BIND_H
#define KPEX_KUDNN_BIND_H

#include <torch/extension.h>
#include "kudnn_adapter.h"

namespace kudnn{
inline void bind(pybind11::module &m)
{
    auto submodule = m.def_submodule("kudnn");
    submodule.def("kudnn_linear", &kudnn_linear, py::arg("input"), py::arg("weight"), py::arg("bias") = py::none());
    submodule.def("kudnn_conv2d", &kudnn_conv2d, py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), 
        py::arg("stride") = py::int_(1), py::arg("padding") = py::int_(0), py::arg("dilation") = py::int_(1), py::arg("groups") = py::int_(1));
    submodule.def("kudnn_conv3d", &kudnn_conv3d, py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), 
        py::arg("stride") = py::int_(1), py::arg("padding") = py::int_(0), py::arg("dilation") = py::int_(1), py::arg("groups") = py::int_(1));
}
}
#endif