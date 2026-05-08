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

model = models.resnet18(pretrained=False) #使用torchvision的resnet18模型, 基于conv2d + linear实现
model = model.eval().to(device)

print("model structure")
print(model)

BATCH_SIZE=6
x = torch.randn(BATCH_SIZE, 3, 224, 224)
x1 = x.clone()

time1 = time.perf_counter()
y = model(x)
time2 = time.perf_counter()
print(f"ResNet18 using origin op , calc total time : {(time2 - time1):.6f}")

optimized_model = kpex.optimize(model)
optimized_model.eval()
time3 = time.perf_counter()
y1 = optimized_model(x1)
time4 = time.perf_counter()
print(f"ResNet18 using optimized op , calc total time : {(time4 - time3):.6f}")

print(f"test optimized model result is close to orignal model result: {torch.allclose(y, y1, atol = 1e-3)}")
