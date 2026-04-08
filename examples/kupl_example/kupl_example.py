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
import torch.distributed as dist
import kpex._C_example as kernel

def kupl_parallel_test():
    out = kernel.kupl_example.test_kupl_parallel()

def kupl_parallel_for_test():
    out = kernel.kupl_example.test_kupl_parallel_for()

def kupl_parallel_test_error():
    out = kernel.kupl_example.test_kupl_parallel_error()

def kupl_parallel_for_test_error():
    out = kernel.kupl_example.test_kupl_parallel_for_error()


