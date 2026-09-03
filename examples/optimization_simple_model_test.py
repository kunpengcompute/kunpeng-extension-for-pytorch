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
import kpex
import torchvision.models as models
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device : {device}")

class Model(torch.nn.Module):
    """Simple model with no parameters - just matrix multiplications."""
    def forward(self, inputs):
        a, b, c, d = inputs
        x1 = torch.mm(a, b)
        x2 = torch.mm(c, d)
        y = torch.mm(x1, x2)
        return y

model = Model()
model = model.eval().to(device)

Size = 400
# For Model, inputs is a tuple of 4 tensors
inputs = (torch.randn(Size, Size), torch.randn(Size, Size),
          torch.randn(Size, Size), torch.randn(Size, Size))
inputs_1 = native_inputs = tuple(t.clone() for t in inputs)

time1 = time.perf_counter()
y = model(inputs)
time2 = time.perf_counter()
print(f"using origin op , calc total time : {(time2 - time1):.6f}")

optimized_model = kpex.optimize(model)
optimized_model.eval()
time3 = time.perf_counter()
y1 = optimized_model(inputs_1)
time4 = time.perf_counter()
print(f"using optimized op , calc total time : {(time4 - time3):.6f}")

print(f"test optimized model result is close to orignal model result: {torch.allclose(y, y1, atol = 1e-3)}")