# torch_kuccl — PyTorch c10d 适配层

PyTorch c10d 适配层：链接 libkuccl.so，把 kuccl 注册为 PyTorch 的 backend。
本包**不含任何 kuccl 库代码**，只是薄适配层（OOB 桥接 + Work 封装 + auto-register）。

## 与 kuccl 库的关系

```
torch_kuccl（本目录）         kuccl（../kuccl，库本体）
  c10d Backend 实现                   纯 C 接口，零 torch
  c10d::BackendKuccl                  纯 C 接口，零 torch
  import kuccl                        MPI 程序可直接链接
  链接 libkuccl.so ──────────────────►
```

## 前置

先构建 kuccl 库（`cd ../kuccl && ./build.sh`），然后（两包独立构建，各管各的）：

```bash
cd torch_kuccl
./build.sh          # 前置检查 + 编译 + import 门禁（环境由 ../kuccl/env.sh 提供）
# 产物：kuccl_backend.so（链接 ../kuccl/build/libkuccl.so）
```

## 使用

```python
import kuccl            # auto-register "kuccl" backend
import torch.distributed as dist

dist.init_process_group("kuccl", rank=0, world_size=1)   # 或 "cpu:kuccl"

x = torch.ones(6)
dist.all_reduce(x)                  # 走 kuccl
```

## 后端自动选择（无需配置）

| 场景 | 自动选择的后端 | 判定依据 |
|------|----------------|----------|
| 单机（torchrun 单节点启动） | `kupl`（shm 快路径） | torchrun 注入的 `LOCAL_WORLD_SIZE == WORLD_SIZE` |
| 跨节点 / 非(torchrun|单机) | `xuccl`（默认） | 其他一切情况 |

优先级：**用户显式 `KUCCL_BACKEND_TYPE` > 自动选择 > 默认 xucg**。
自动选中的 kupl 不可用时（未编入/初始化失败）自动回退 xucg 并 TORCH_WARN 提示。
手动 A/B：`KUCCL_BACKEND_TYPE=kupl python tests/correctness.py ...`。
（kupl 需 `KUPL_DIR` 指向安装前缀重新编译 libkuccl，见 ../kuccl/README.md。）

## 算子支持矩阵

| c10d 算子 | 实现 | 说明 |
|-----------|------|------|
| all_reduce / allreduce_coalesced | `kucclAllReduce` | 多 tensor 自动拼桶（1 轮通信） |
| all_gather / all_gather_into_tensor | `kucclAllGather` | |
| broadcast | `kucclBroadcast` | |
| all_to_all（等长 split） | `kucclAllToAll` | 零拷贝直通 |
| all_to_all（变长 split） | 不支持 | 显式报错；需要变长请用 gloo |
| barrier | 1 字节 allreduce | `KUCCL_BARRIER=store` 回退 TCPStore |
| reduce / reduce_scatter / gather / scatter | 未实现 | c10d 默认报错 |
| send / recv | 不支持 | 公开面收敛为 4 个集合通信接口 |

异步语义：所有算子返回 `WorkKuccl`（真异步，V2.4）——`async_op=False` 时
c10d 调 `work.wait()`；`async_op=True` 时 Work 交给用户，轮询循环在
`wait()/isCompleted()` 内部。

## 目录结构

```
torch_kuccl/
├── include/           backend.hpp（WorkKuccl/BackendKuccl）、oob.hpp
├── csrc/              backend.cpp、oob.cpp（Store 版 OOB）
├── kuccl/             Python 包（预加载 HUCX/XUCG + auto-register）
├── tests/             correctness.py（功能门禁）+ test_*.py（pytest 套件）
├── examples/          benchmark.py（性能对比）、simple_allgather.py 等
├── benchmarks/        DeepSpeed communication 套件（compare_run.py 等）
├── multinode/         run_all.sh / run_node.sh 多节点编排
├── test.sh            单机功能测试一键跑
└── setup.py
```

## 多节点

```bash
NNODES=2 PPN=2 bash multinode/run_all.sh     # 冒烟
NNODES=8 PPN=16 bash multinode/run_all.sh    # 设计书验收口径
BACKEND=gloo bash multinode/run_all.sh       # gloo 基线
bash multinode/run_all.sh compare            # 30% 判定
```

TLS 策略（评审约定）：单机 `sm,self`；跨节点 `rc,self`（ud 备选 `ud,self`，
跨节点配置不含 sm）——由 `run_node.sh` 按 NNODES 自动设置。
