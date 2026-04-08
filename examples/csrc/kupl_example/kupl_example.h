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

#ifndef KPEX_EXAMPLE_KUPL_EXAMPLE_H
#define KPEX_EXAMPLE_KUPL_EXAMPLE_H

#include <ATen/core/Tensor.h>

namespace kupl_example {

void test_kupl_parallel();
void test_kupl_parallel_for();
void test_kupl_parallel_error();
void test_kupl_parallel_for_error();

}   // namespace kupl_example

#endif
