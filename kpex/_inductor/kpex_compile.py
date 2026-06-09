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

import os
import torch
from torch._inductor.compile_fx import compile_fx
from .kpex_fusion import kpex_fusion_passes

def kpex_compile(gm, example_inputs):
    if os.getenv("KPEX_ENABLE_KUDNN_BACKEND", "false").lower() == "true":
        kpex_fusion_passes(gm, example_inputs) # Register extra fusion graph passes
    return compile_fx(gm, example_inputs) # Return to the native inductor graph compile passes