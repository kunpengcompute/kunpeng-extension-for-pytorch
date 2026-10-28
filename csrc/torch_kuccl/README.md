# torch_kuccl — PyTorch 适配层

把 kuccl 注册为 PyTorch backend 的薄适配层（OOB 桥接 + Work 封装 + auto-register），
本包不含 kuccl 库代码，链接 `../kuccl/build/libkuccl.so`。

## 环境准备

前置：鲲鹏节点 + HPCKit（含 HUCX / XUCG / hmpi）。

```bash
conda create -n kuccl python=3.12 -y
conda activate kuccl
pip install torch==2.9.1 numpy
```

## 构建与测试

```bash
cd csrc/kuccl
source kuccl_env.sh                # 库路径 + 编译器（路径不对时先 export HPCKIT=<路径>）
./build.sh clean && ./build.sh

cd ../torch_kuccl
source env.sh                      # conda + TORCH_LIB
./build.sh
pip install -e . --no-build-isolation

python tests/correctness.py --world-size 2

# 回归矩阵：ws × 3 遍（挂死也判 FAIL，全绿才退出 0）
./tests/qa_matrix.sh

# 长稳模式（ws=16 大轮次）
ITERS=200 WORLD_SIZES=16 ROUNDS=1 ./tests/qa_matrix.sh
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

## 已知限制（kupl 后端）

- 同步驱动：算子内部死锁不受看门狗保护（完整清单见 `../kuccl/README.md`
  「支持矩阵与已知限制」）
- MIN/MAX 为本地模拟（KUPL 底层仅原生 SUM），功能语义与原生一致
- all_to_all 仅等长；变长请用 gloo
- 排障第一步：适配层启动会打
  `[KuCCL] init: kuccl <版本> (backend=..., ws=...)`——先核对版本与
  后端是否符合预期，再谈别的

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `KUCCL_BACKEND_TYPE` | 自动 | `xucg`（默认）/ `kupl`（单机 shm） |
| `KUCCL_TOPOLOGY` | 开 | 进程拓扑喂入（node_id / NUMA），`=0` 回退平铺 |
| `KUCCL_LOG_LEVEL` | WARN | 库日志级别 |
| `KUCCL_LOG_FILE` | 空 | 日志落盘路径（`%r` = rank 分文件；跑分后查 TIMING/告警用） |
| `KUCCL_KUPL_TIMING` | 关 | kupl 后端分阶段耗时汇总（comm 销毁时输出） |
| `KUCCL_KUPL_AG_ALGO` | 不设=linear | allgather 算法：`linear`（默认，线性直读）/ `auto` / `hier`（拓扑合适→分层，否则 linear，粒度自动选）/ `hier_die` / `hier_socket`（显式粒度，A/B 用）/ `smart`（按大小桶学习） |
| `KUCCL_KUPL_AG_SYNC` | fence | linear 路径同步原语：`fence` / `doorbell` |
| `KUCCL_KUPL_AG_THREADS` | 4 | allgather 大消息并行读线程数（=1 关并行） |
| `KUCCL_KUPL_AG_PAR_MIN` | 16KB | 并行读触发的最小消息尺寸 |
| `KUCCL_KUPL_AR_ALGO` | 不设=flat | allreduce 算法：`flat`（默认，扁平路径：SUM 走 KUPL 原生）/ `hier`（分层，粒度按拓扑选）/ `auto`（≥AR_HIER_MIN 走 hier，否则 flat） |
| `KUCCL_KUPL_AR_HIER_MIN` | 512KB | auto 路由切分层的尺寸阈值 |
| `KUCCL_KUPL_BC_ALGO` | direct | broadcast 算法：`direct`（门铃直读，零窗口税）/`native`（KUPL 原生，A/B 用） |
| `KUCCL_KUPL_SPIN_MAX` | 200000 | 门铃/ack 等待的 yield 前自旋次数（核共享场景调小） |
| `TORCH_KUCCL_ALLREDUCE_INPLACE` | 关 | allreduce 原地模式 |
| `TORCH_KUCCL_FLUSH` | 关 | 每轮集合后 flush |
| `TORCH_KUCCL_BARRIER` | 空 | `store` 时 barrier 走 TCPStore |

## 单节点测试（kupl 后端）

PyTorch 官方 CPU collective 方法体原样复用（16 个，固定提交，来源/行号/SHA256
见 `benchmarks/communication/PYTORCH_CPU_TESTS.md`），只替换启动与断言框架：
每 rank 子进程跑官方方法体，参数注入切出单个（算子 × dtype × count ×
sync/async）组合；每用例用独立 TCPStore 汇总全 rank 结果——任何 rank 失败、
结果缺失、崩溃、超时都判 FAIL，不用被测的集合通信自证通过。

```bash
cd benchmarks/communication

# 冒烟
python test_pytorch_collectives.py --suite smoke --profiles kupl-default --world-sizes 16

# 全量矩阵：full × plain/doorbell × 串行/并行读
python test_pytorch_collectives.py --suite full \
    --profiles kupl-default kupl-doorbell --world-sizes 16 --threads 1 4

# 分层路径 A/B（0 8 换成 lscpu 实际 NUMA 节点号）
python test_pytorch_collectives.py --suite full --profiles kupl-hier \
    --world-sizes 16 --numa-nodes 0 8 --require-hier
```

退出码非 0 即有 FAIL；结果与各 rank 日志在 `pytorch_cpu_results/<run>/`。
覆盖与判定细则见 `PYTORCH_CPU_TESTS.md`。

## 单节点性能（kupl vs gloo）

DeepSpeed 通信套件口径，同参数分别跑 `--backend kuccl` 与 `--backend gloo`，
逐尺寸输出 latency/busbw 对比表；平均 latency 提升 ≥30% 时退出码 0。

```bash
cd benchmarks/communication

# ws=16，8B~1MB 扫描（maxsize 计元素个数，float32 ×4B）
python compare_run.py --ops all_reduce all_gather broadcast all_to_all \
    --world-size 16 --maxsize 19 --trials 30 --warmups 5
```

规模扩展性（延迟随 ws 增长曲线，出 csv + 图，需 matplotlib）：

```bash
python ws_scaling.py --world-sizes 2 4 8 16    # --skip-gloo 只跑 kupl
```

消息规模（`--maxsize`，2 的幂）、算子与透传参数见 `--help`。
C ABI 口径的 MPI 直调基准（bench_ag_mpi.sh 等）在 `../kuccl/examples/`，见 kuccl README「基准」。

## 多节点测试

```bash
cd benchmarks/communication

# 1) 正确性（先跑这个，通过才有性能意义）
python compare_nodes.py --nnodes 8 --ppn 16 \
    --ops all_reduce,all_gather,broadcast,all_to_all \
    --gate --backends kuccl --local

# 2) 性能对比（kuccl vs gloo，输出逐尺寸延迟表与判定）
python compare_nodes.py --nnodes 8 --ppn 16 \
    --ops all_reduce,all_gather,broadcast,all_to_all \
    --trials 20 --maxsize 19 --local
```

性能轮自动先跑 kuccl 再跑 gloo，末尾输出逐尺寸对比表与判定
（平均延迟提升 ≥30%）。节点清单与 TLS 在 `config/default.py`。

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
└── tests/             correctness.py + pytest 套件
```

基准与 PyTorch 测试套件独立在仓库根 `benchmarks/`（与 csrc 同级）。
