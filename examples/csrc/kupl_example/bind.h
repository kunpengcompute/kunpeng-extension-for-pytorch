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

#ifndef KPEX_EXAMPLE_BIND_H
#define KPEX_EXAMPLE_BIND_H

#include <torch/extension.h>

#include "kupl_example.h"

namespace kupl_example {
inline void bind(pybind11::module &m)
{
    auto submodule = m.def_submodule("kupl_example");
    submodule.def("test_kupl_parallel", &test_kupl_parallel);
    submodule.def("test_kupl_parallel_for", &test_kupl_parallel_for);
    submodule.def("test_kupl_parallel_error", &test_kupl_parallel_error);
    submodule.def("test_kupl_parallel_for_error", &test_kupl_parallel_for_error);
}
}

#endif
