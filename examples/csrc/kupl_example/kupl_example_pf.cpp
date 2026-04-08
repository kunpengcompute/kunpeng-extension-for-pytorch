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

#include <iostream>
#include <unistd.h>
#include <cassert>
#include <atomic>
#include "utils/parallel.h"
#include "kupl_example.h"

namespace kupl_example {
void test_kupl_parallel()
{
    std::cout << "test_kupl_parallel " << std::endl;
    int num_threads = kupl_get_num_executors();
    int tid = kupl_get_executor_num();
    kpex::parallel(num_threads, [&](int tid) {
        std::cout << "KUPL Thread " << tid << std::endl;
    });
}


void test_kupl_parallel_for()
{
    std::cout << "test_kupl_parallel_for " << std::endl;
    std::atomic<int> count = {0};
    kpex::parallel_for(0, 1024, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            count.fetch_add(1);
        }
    });
    if (count.load() == 1024) {
        std::cout << "test_kupl_parallel_for PASS!" << std::endl;
        return;
    }
    std::cout << "test_kupl_parallel_for ERROR!" << std::endl;
}

void test_kupl_parallel_error()
{
    std::cout << "test_kupl_parallel " << std::endl;
    int num_threads = -2;
    int tid = kupl_get_executor_num();
    kpex::parallel(num_threads, [&](int tid) {
        std::cout << "KUPL Thread " << tid << std::endl;
    });
}

void test_kupl_parallel_for_error()
{
    std::cout << "test_kupl_parallel_for " << std::endl;
    int out[1024] = {0};
    kpex::parallel_for(1024, 0, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            out[i] = i * i;
        }
    });
}

} // namespace kupl_example
