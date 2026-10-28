# 用 PyTorch 官方 CPU collective 用例测试 KuCCL

入口 `test_pytorch_collectives.py`：先 `import kuccl` 注册后端，再
`dist.init_process_group(backend="kuccl", ...)`。`kupl-default`、`kupl-doorbell`、
`kupl-hier`、`xucg` 都走 `kuccl`，只切底层实现；`gloo` 仅开发期对照。

单机多进程正确性测试，不测带宽/时延，直接 `python` 启动（不要套 torchrun、
mpirun 或 run_cpu.py）。调用走 PyTorch 适配层，不等同于 C ABI 单元测试。

## 官方方法体来源

PyTorch 仓库相对路径，链接固定到具体版本：

| 完整路径 | 用途 |
| --- | --- |
| [`test/distributed/test_c10d_collectives.py`](https://github.com/pytorch/pytorch/blob/d0b9d245f4f400496b3a41355330b4322ca267df/test/distributed/test_c10d_collectives.py) | 新版后端通用 collective 矩阵；复用其中四个 collective 的测试方法与辅助方法 |
| [`test/distributed/c10d_backend_common.py`](https://github.com/pytorch/pytorch/blob/d0b9d245f4f400496b3a41355330b4322ca267df/test/distributed/c10d_backend_common.py) | 新版测试的后端实例化、进程与能力定义；启动框架由本地 harness 替代 |
| [`test/distributed/test_c10d_gloo.py`](https://github.com/pytorch/pytorch/blob/v2.9.0/test/distributed/test_c10d_gloo.py) | Gloo 专用用例（basics/stress/参数检查/coalesced） |
| [`test/distributed/test_c10d_spawn_gloo.py`](https://github.com/pytorch/pytorch/blob/v2.9.0/test/distributed/test_c10d_spawn_gloo.py) | CPU spawn 用例：`test_broadcast/test_all_gather/test_all_to_all_single` |
| [`test/distributed/test_c10d_spawn.py`](https://github.com/pytorch/pytorch/blob/v2.9.0/test/distributed/test_c10d_spawn.py) | spawn 测试共用实现，供上一文件调用 |
| [`test/distributed/test_distributed_spawn.py`](https://github.com/pytorch/pytorch/blob/v2.9.0/test/distributed/test_distributed_spawn.py) | 通用 distributed 测试的多进程入口 |
| [`torch/testing/_internal/distributed/distributed_test.py`](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/testing/_internal/distributed/distributed_test.py) | 2.9 通用实现：`test_all_reduce_sum/min/max`、`test_broadcast`、`test_all_gather`、`test_all_to_all_single_equal_split` 及 group/full_group 变体；同文件含 CUDA 用例 |

`test_c10d_collectives.py` 只在 main 上有，v2.9.0 没有。无需下载编译 PyTorch：
16 个官方方法体原样内置在 `pytorch_cpu_upstream.py`（固定提交
`d0b9d245f4f400496b3a41355330b4322ca267df`），来源路径、行号、SHA256 在
`pytorch_cpu_upstream.json`，许可在 `pytorch_cpu_upstream.LICENSE`。
只换启动/断言框架和能力矩阵，方法体不动：

| 被测实现 | 实际调用的官方方法 |
| --- | --- |
| AllReduce | `test_all_reduce`（SUM/MIN/MAX） |
| Broadcast | `test_broadcast` |
| AllGather | `test_all_gather`（list 输出）、`test_all_gather_single`（连续 tensor） |
| AllToAll | `test_all_to_all_single`（等长交换） |

旧版 PyTorch 没有 `all_gather_single` 时映射到公开 API `all_gather_into_tensor`。
不改全局 `torch.distributed`，不依赖 `torch.testing._internal`。

## 运行

编译好 libkuccl 与当前 PyTorch 对应的扩展后，从仓库根目录：

```bash
source csrc/torch_kuccl/env.sh
source csrc/kuccl/kuccl_env.sh
cd benchmarks/communication

# 冒烟：四算子、同步/异步、结果校验
python test_pytorch_collectives.py --suite smoke --profiles kupl-default --world-sizes 2

# 全量：KuPL plain/doorbell 与 XUCG，含非 2 次幂 rank 数
python test_pytorch_collectives.py --suite full \
  --profiles kupl-default kupl-doorbell xucg \
  --world-sizes 1 2 3 4 5 8 16 --threads 1 4 \
  --repeats 5 --max-memory-mib 512
```

环境激活不能替代扩展编译，升级 PyTorch 或改 C++ 后先重编译。程序优先导入
当前 checkout 的 `csrc/torch_kuccl/kuccl`，导入/加载/注册失败记 FAIL、退出非零。

分层算法单独测。先用 `lscpu -e=CPU,NODE,SOCKET` 查 NUMA 节点，
下面的 `0 8` 换成实际节点号：

```bash
python test_pytorch_collectives.py --suite full --profiles kupl-hier \
  --world-sizes 4 5 8 16 --threads 1 4 --numa-nodes 0 8 \
  --require-hier --repeats 5 --max-memory-mib 512
```

rank 按节点轮转分配（5 ranks 即 3+2），每 rank 可用所在节点的全部 CPU
（不独占、非内存绑定）。两节点各 ≥2 ranks 才成层次拓扑，2/3 ranks 回退
linear；同 socket 两节点走 die 分层，跨 socket 走 socket 分层。

程序开 KuCCL TRACE，只打各 communicator 的 rank 0，日志留在结果目录。
`observed_routes` 取自实际 `[AR-route]`/`[AG-route]` 日志并按 world size 筛选；
`--require-hier` 要求 AR 与 AG 都观察到 `hier_die/hier_socket`，
且 AR、AG 一起跑。数据通过与路径覆盖分开记录，缺证据提示 COVERAGE，
加 `--require-hier` 判 FAIL。

native 分支与 AllReduce 原地分支的补充命令：

```bash
python test_pytorch_collectives.py --suite full --profiles kupl-default \
  --world-sizes 2 4 8 --a2a-algo native --bc-algo native --max-memory-mib 512

python test_pytorch_collectives.py --suite full --profiles kupl-default kupl-hier \
  --world-sizes 4 5 8 --threads 1 4 --numa-nodes 0 8 \
  --ar-inplace 1 --max-memory-mib 512
```

`--threads` 即 `KUCCL_KUPL_AG_THREADS`，是否真并行由算法、rank 数和拓扑决定。
`--torch-threads` 单独控制 OMP/MKL（默认 1）。XUCG 每个 world size 只跑一次。

## 覆盖与判定

- 官方正向矩阵：CPU `int8/uint8/int32/int64/float32/float64`，SUM/MIN/MAX，
  同步与 async+wait，count 取 `0,1,4,17,4097,16384,131073,262145`
  （AllToAll 为每 peer 元素数）。
- 扩展正确性：数据由元素、源 rank、目标 rank、轮次共同构造；重复跑四个算子；
  MIN/MAX 逐元素获胜 rank 变化；全部 broadcast roots；二维 AllToAll 与显式等长
  row splits；buffer 先增后缩；非连续 AllGather/AllToAll 输入归一化。
- communicator：全员新 group、奇偶子组、相同成员重复建 group，各跑四个算子，
  同步/异步。
- coalesced AllReduce：多个不同大小 tensor；2.9 的
  `all_reduce_coalesced(async_op=True)` 返回 Future，单独等待并检查。
- AllToAll 走 `kucclUint8` 传原始字节，另测
  `int16/float16/bfloat16/bool/complex64/complex128`；其余三个算子不含这些
  dtype 承诺。
- 负向契约：拒绝 PRODUCT、变长 AllToAll splits、非连续 AllToAll 输出，
  负向 PASS 指按约定拒绝；单 rank 无法构造变长 peer splits，SKIP。

范围外：list 形式 `all_to_all`、uneven AllGather、reduce_scatter、
gather/scatter、send/recv、多机通信、任意多并发未完成 Work；异步用例均显式等待。

空 tensor 是官方正向用例，正常判定。失败按 `case.id`、rank traceback 和 KuCCL
日志定位。某 rank 提前失败时 watchdog 终止本组，未跑用例标 NOT_RUN。

每用例用独立 TCPStore 汇总全 rank 结果，不用被测集合自证通过。任何 rank
失败、结果缺失/重复、崩溃、超时都判非零退出。assertion 失败可继续时继续
收集；卡住时 kill 本组 rank 进程，保留已有结果继续后续配置。每用例/初始化
默认 120 秒，watchdog 另加 15 秒，慢机用 `--timeout` 调大。

`--max-memory-mib` 按 rank tensor 峰值保守估算（不含库内 scratch、共享窗口与
进程内存），超预算 SKIP，确认矩阵跑完要看 SKIP。world size 限 1..16。

## 看结果与复现

```text
RUN kupl-default-ws2-threads4
PASS kupl-default-ws2-threads4: {'PASS': ...}
```

结果在 `pytorch_cpu_results/<UTC时间>/`，每次新建：

```text
manifest.json                         # 参数、Python 版本、官方源码 provenance
summary.json                          # 每个 profile/world-size/threads 的聚合
kupl-default-ws2-threads4/
  report.json                         # 所有用例与所有 rank 的判定、错误、实际路由
  cases.jsonl                         # 每个已完成用例即时落盘
  rank-0.log, rank-1.log, ...         # 含 C++ stdout/stderr 的完整日志
  rank-0.metadata.json, ...           # torch 版本、KuCCL 导入位置、绑定、实际环境
  progress.json                       # watchdog 最近看到的用例编号
```

只看计划、不导入 torch：

```bash
python test_pytorch_collectives.py --suite full --world-sizes 4 --list
```

按失败 ID 片段复现：

```bash
python test_pytorch_collectives.py --suite full --profiles kupl-default \
  --world-sizes 4 --case-filter 'pytorch/alltoall/alltoall/float32/n17'
```

本机回归不依赖 PyTorch/扩展：

```bash
python -m unittest discover -s csrc/torch_kuccl/tests -p test_pytorch_cpu_runner.py -v
```

覆盖方法体完整性、矩阵、全 rank 判定、损坏日志回收、路由识别、崩溃/超时清理。
harness 已用 PyTorch 2.9 CPU Gloo 验证；KuPL/XUCG 实机结果以 kuccl 命令为准，
Gloo 对照跳过 KuCCL 专属用例（字节传输、非连续 A2A 输入、拒绝契约）。
