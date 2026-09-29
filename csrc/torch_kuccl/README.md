# torch_kuccl — PyTorch 适配层

把 kuccl 注册为 PyTorch backend 的薄适配层（OOB 桥接 + Work 封装 + auto-register），
本包不含 kuccl 库代码，链接 `../kuccl/build/libkuccl.so`。

## 环境准备

前置：鲲鹏节点 + HPCKit（含 HUCX / XUCG / hmpi）。

```bash
# 1) Python 环境
conda create -n kuccl python=3.12 -y
conda activate kuccl

# 2) 安装 PyTorch（CPU 版）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3) 库环境（两个脚本独立，都要 source）
cd csrc/kuccl      && source kuccl_env.sh    # HPCKit 路径，不对时先 export HPCKIT=<路径>
cd ../torch_kuccl  && source env.sh          # conda + TORCH_LIB（环境名不同时先 export PYTORCH_ENV=<名>）
```

## 构建与测试

```bash
cd csrc/kuccl      && ./build.sh     # 产物 build/libkuccl.so
cd ../torch_kuccl  && ./build.sh     # 产物 kuccl_backend.so

bash test.sh                          # 单机功能一键跑
python tests/correctness.py --world-size 2
```

## 使用

```python
import kuccl                                  # auto-register "kuccl" backend
import torch.distributed as dist

dist.init_process_group("kuccl", rank=0, world_size=1)

x = torch.ones(6)
dist.all_reduce(x)                            # 走 kuccl
```

## 算子支持

| c10d 算子 | 实现 |
|-----------|------|
| all_reduce（含 coalesced 拼桶） | `kucclAllReduce` |
| all_gather / all_gather_into_tensor | `kucclAllGather` |
| broadcast | `kucclBroadcast` |
| all_to_all（等长） | `kucclAllToAll` |
| barrier | 1 字节 allreduce（`TORCH_KUCCL_BARRIER=store` 回退 TCPStore） |
| 变长 all_to_all / reduce / send / recv 等 | 不支持（用 gloo） |

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `KUCCL_BACKEND_TYPE` | 自动 | `xucg`（默认）/ `kupl`（单机 shm） |
| `KUCCL_TOPOLOGY` | 开 | 进程拓扑喂入（node_id / NUMA），`=0` 回退平铺 |
| `KUCCL_LOG_LEVEL` | WARN | 库日志级别 |
| `TORCH_KUCCL_ALLREDUCE_INPLACE` | 关 | allreduce 原地模式 |
| `TORCH_KUCCL_FLUSH` | 关 | 每轮集合后 flush |
| `TORCH_KUCCL_BARRIER` | 空 | `store` 时 barrier 走 TCPStore |

## 多节点测试

```bash
cd benchmarks/communication

# 1) 正确性门禁（先跑这个，通过才有性能意义）
python compare_nodes.py --nnodes 8 --ppn 16 \
    --ops all_reduce,all_gather,broadcast,all_to_all \
    --gate --backends kuccl --local

# 2) 性能对比（kuccl vs gloo，输出逐尺寸延迟表与验收判定）
python compare_nodes.py --nnodes 8 --ppn 16 \
    --ops all_reduce,all_gather,broadcast,all_to_all \
    --trials 20 --maxsize 19 --local
```

性能轮自动先跑 kuccl 再跑 gloo，末尾输出逐尺寸对比表与判定
（平均延迟提升 ≥30% 为达标）。节点清单与 TLS 在 `config/default.py`。

查看每档消息实际选中的算法（含 node-aware 路由）：

```bash
python plan_map.py
```

其余参数（单算子、预热轮数、环境变量透传等）见 `--help`。

## 目录结构

```
torch_kuccl/
├── csrc/ include/     适配层（BackendKuccl / OOB）
├── kuccl/             Python 包（预加载 HUCX/XUCG + auto-register）
├── env.sh             conda + TORCH_LIB 激活
├── build.sh / test.sh / setup.py
├── tests/             correctness.py + pytest 套件
└── benchmarks/        通信基准套件（compare_nodes.py 多节点编排）
```
