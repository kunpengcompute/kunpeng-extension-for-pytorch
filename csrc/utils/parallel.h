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

#ifndef KPEX_UTILS_PARALLEL_H
#define KPEX_UTILS_PARALLEL_H

#include <algorithm>
#include <atomic>
#include <iostream>
#include "kupl.h"


namespace kpex {
namespace {
inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

template <typename F>
struct parallel_for_function_args {
    int64_t begin;
    int64_t end;
    int64_t grain_size;
    const F& f;
};

template <typename F>
static void parallel_for_function(kupl_nd_range_t *nd_range, void *args, int tid, int tnum)
{
    auto data = (parallel_for_function_args<F> *)args;
    const F& f = data->f;
    f(nd_range->nd_range[0].lower, nd_range->nd_range[0].upper);
}

template <typename F>
struct parallel_function_args {
    const F& f;
};

template <typename F>
static void parallel_function(kupl_nd_range_t *nd_range, void *args, int tid, int tnum)
{
    auto data = (parallel_for_function_args<F> *)args;
    const F& f = data->f;
    f(tid);
}
}

template <typename F>
inline void parallel_for(int64_t begin, int64_t end, int64_t grain_size, const F& f) {
    int num_executors = kupl_get_num_executors();
    kupl_nd_range_t range;
    KUPL_STRIDE_1D_RANGE_INIT(range, begin, end, 1, grain_size);
    kupl_parallel_for_desc_t desc = {
        .field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
        .range = &range,
        .egroup = NULL,
        .concurrency = num_executors,
        .policy = KUPL_LOOP_POLICY_STATIC
    };
    parallel_for_function_args<F> args = {begin, end, grain_size, f};
    int ret = kupl_parallel_for(&desc, parallel_for_function<F>, &args);
    if (ret == KUPL_ERROR) {
        std::cout << "Error due to parallel_for function." << std::endl;
    }
}

template <typename F>
inline void parallel(int num_threads, const F& f) {
    kupl_parallel_for_desc_t desc = {
        .field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
        .range = NULL,
        .egroup = NULL,
        .concurrency = num_threads,
        .policy = KUPL_LOOP_POLICY_STATIC
    };
    parallel_function_args<F> args = {f};
    int ret = kupl_parallel_for(&desc, parallel_function<F>, &args);
    if (ret == KUPL_ERROR) {
        std::cout << "Error due to parallel function." << std::endl;
    }
}

} // kpex

#endif
