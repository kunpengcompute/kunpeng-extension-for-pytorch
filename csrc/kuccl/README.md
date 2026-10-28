# kuccl — 鲲鹏集合通信库

基于 UCG 公有接口的纯 C 集合通信库（`libkuccl.so`），零框架依赖。
PyTorch 接入见 [torch_kuccl](../torch_kuccl/README.md)。

## 目录结构

```
kuccl/
├── include/          公开头文件（kuccl.h / types.hpp）
├── src/              实现（coll / comm / log / error + backend/{xucg,kupl}）
├── examples/         coll_test_mpi.cpp（正确性）+ 基准脚本（见下）
├── tests/            test_a2a_bench.py（a2a 基准判定的主机回归，不依赖 KUPL）
├── kuccl_env.sh      库环境一键就绪（HPCKit/HUCX/XUCG + MPI 工具链）
└── build.sh / CMakeLists.txt / kuccl.map
```

## 环境与构建

前置：HPCKit（含 HUCX / XUCG / hmpi）。默认路径在 `kuccl_env.sh`，可外部覆盖：

```bash
source ./kuccl_env.sh            # 库路径 + MPI 工具链（PATH/OPAL_PREFIX）一键就绪
                                  # 安装路径非默认时：先 export HPCKIT=<实际路径>
source ./kuccl_env.sh --check    # 可选：逐项自检（库/编译器/mpicxx 探活）
./build.sh                       # 产物：build/libkuccl.so
```

## MPI 直调用例

直调 `libkuccl` C 接口的最小正确性入口：MPI 只负责拉起进程和提供 OOB
全交换（`kucclOobAllgatherFn` 回调注入），集合本身走 kuccl。覆盖四个算子
（各带逐元素数据校验）、异步 `kucclTest` 轮询、`kucclCommDestroy`；
`--test-split` 附加 `kucclCommSplit` 子通信域及其上的 AllReduce。

```bash
source ../kuccl_env.sh

mpicxx coll_test_mpi.cpp -I../include -L../build -lkuccl -o coll_test_mpi

# 单节点（ppn=16）
mpirun --allow-run-as-root -np 16 --mca coll ^ucg ./coll_test_mpi

# 多节点
cat > hosts.txt <<EOF
<节点IP1> slots=16
<节点IP2> slots=16
EOF
mpirun --allow-run-as-root -np 32 --hostfile hosts.txt \
    -x LD_LIBRARY_PATH -x OPAL_PREFIX -x UCX_TLS \
    --mca coll ^ucg \
    /绝对路径/csrc/kuccl/examples/coll_test_mpi
```

预期：AllReduce / Broadcast / AllGather / AllToAll 四项 `PASS`；
多节点时 `KUCCL_LOG_LEVEL=INFO` 可见 `topology(src=host-hash): nodes=<N> ppn=<slots>`。

参数说明：`--mca coll ^ucg` 排除 hmpi 自带的 UCG 集合组件（避免与 kuccl 的 UCG
实例冲突）；`-x` 向远端进程转发环境变量（跨节点必须，单节点经 fork 自然继承）；
多节点必须用绝对路径（远端进程工作目录不同）。

## 对外接口（kuccl.h）

| 类别 | 接口 |
|------|------|
| 通信域 | `kucclGetUniqueId` `kucclCommInitRankConfig` `kucclCommSplit` `kucclCommDestroy` `kucclCommAbort` |
| 集合算子 | `kucclAllReduce` `kucclAllGather` `kucclBroadcast` `kucclAllToAll` |
| 异步 / 辅助 | `kucclTest` `kucclGetLastError` `kucclFlushComm` |
| 版本查询 | `kucclGetVersion`（KuCCL_1.2 节点，排障第一步先核对版本） |

AllToAllV 与点对点不在公开面。

## 后端

| 后端 | 状态 | 说明 |
|------|------|------|
| `xucg` | 默认 | UCG 公有接口，单机 + 跨节点 |
| `kupl` | 单机 | KUPL shm 快路径，需 `KUPL_DIR` 指向安装前缀重新编译 |

选择：环境变量 `KUCCL_BACKEND_TYPE` 或 `kucclConfig_t.backend`。

## xucg 后端

UCG 公有接口（`ucg_request_*`）的薄封装，算法与路由交给 UCG 原生实现，
单机跨节点同一套代码。相关旋钮：

- `KUCCL_TOPOLOGY`（默认开）：主机名哈希 OOB 交换喂 node_id 给 UCG，
  节点感知算法的输入；`=0` 回退平铺（A/B 对照用）
- `KUCCL_VDT`：bcast / allgather / alltoall 的 dtype 语义。`bytes`（默认）=
  UINT8 + 字节数，`elem` = 真实 dtype + 元素数，两种语义下 UCG plan 选择
  不同；allreduce 不受影响（归约必须真实 dtype）
- `KUCCL_PERSISTENT_REQ=1`：同 key 集合（AllReduce/Broadcast）复用持久化
  请求，免 init 快路径（默认关）
- `KUCCL_PROFILE=1`：分阶段耗时统计
- MIN/MAX 为原生 UCG 归约，无模拟路径

## kupl 后端（单机 shm）

### allgather

默认（不设 `KUCCL_KUPL_AG_ALGO`）走 linear 线性直读：全员互读直达用户缓冲，
流量踩在 allgather 下界上，大块读（≥16KB）自动并行发车。
`auto` 与 `hier` 按拓扑路由：有分层边界用分层配对（socket 优先、die 次之），
同 die 退回 linear。

分层配对（hybrid pair：本组直读 + 跨组搭档搬运 + 落地收账）的选项：

| 强制值 | 算法 | 适用 |
|------|------|------|
| `hier` | 分层配对（粒度按拓扑自动选）| 显式声明与 auto 相同的分层路由：跨 socket→hier_socket，同 socket 跨 die→hier_die，同 die→linear |
| `hier_die` | die 边界配对 | 显式 die 粒度（A/B 测量用）|
| `hier_socket` | socket 边界配对 | 显式 socket 粒度（A/B 测量用）|

- 分布不均时配对自动松弛（搭档挂到远端组末位），某 die 不足 2 rank 自动退回 linear
- `KUCCL_KUPL_AG_ALGO=smart` 按消息大小桶实测学习（worst-rank 投票定案）
- `KUCCL_KUPL_AG_SYNC=fence/doorbell` 选择 linear 路径的同步原语（默认 fence；
  doorbell 用 per-peer 原子门铃替代全组栅栏，小消息更快）

### allreduce

对外只有两个算法，`KUCCL_KUPL_AR_ALGO = flat | hier`（不设 = flat，`auto` 按尺寸路由）：

| 值 | 算法 |
|------|------|
| `flat`（默认）| 扁平路径：SUM→KUPL 原生递归折半/倍增，MIN/MAX→自研 RH-RD（二次幂 N>4）或直读 |
| `hier` | 分层归约（组内 reduce-scatter → **o/e 子块交换** → 属主扇出），**粒度按拓扑自动选**：跨 socket 布局→socket 组、同 socket 跨 die→die 组、同 die→自动退回 flat（[AR-route] 日志可见实际选择） |

- **o/e 子块互补**：切片再切 K 个子块，子块 j 由 j 组属主
  计算全局并**原地折叠**——每个属主只读远端"自己那个子块"，我写的格子无人
  并发读，交换竞态按索引不相交消失（recv 段 1×）
- ③ 属主收拢外组子块后扇出全组（每切片每组唯一写者）；**每元素全局只算一遍**
- 跨组流量：每份数据只以"已归约"形态过一次组边界，属主侧合计
  `2(K−1)/K × count/M`（K=2 与双区持平，K>2 减半）；同步为 **1 次全组 fence
  （尾部 recycle）+ 3 轮 per-partner 门铃**（stage→①→②→③；扁平 RH-RD 为
  `2+2·log₂N` 次全组栅栏）
- **条带并行折叠**：①② 按 dst 不相交
  条带扇出到 KUPL executor（≥16KB 自动并行，之下串行回退），无硬编码线程数、
  无 N 整除约束；③ fetch 串行（K−1 整块拷贝）+ 扇出并行
- 切片数 M = 全局最小组大小：分布不均时大组多余成员不持切片（仍接收 ③ 扇出），
  组间交换按索引天然对齐；某组不足 2 rank 自动退回 plain
- SUM 的 float/double 归约顺序与 KUPL 原生不同——不保证逐位相等（容差内正确）

### alltoall

`KUCCL_KUPL_A2A_ALGO = direct | native`（不设或 `auto` = direct）：

| 值 | 算法 |
|------|------|
| `direct`（默认）| 直拉：一次 staging + 头尾两个 fence，各 rank 从组友 send 段直拉自己的块直写用户 recvbuf——无原生调用、无拷回 |
| `native` | KUPL 原生 alltoall：staging → 原生 → 拷回全量。封装默认设 `KUPL_SHM_ALLTOALL_ALGORITHM=1`，显式设置该变量可覆盖 |

`bench_a2a_mpi.sh` 使用含来源 rank、目的 rank、元素下标的 int32 数据，
warmup 后与计时后各汇总一次全 rank 校验；warmup=0 时也执行一次不计时的
正确性检查。不依赖 KUPL 的主机回归（验证数据布局与脚本判定，需 g++ + Bash）：

```bash
python -m unittest discover -s csrc/kuccl/tests -p test_a2a_bench.py -v
```

该检查不能替代目标机器上的 MPI/KUPL、共享映射、同步与 SDMA 集成测试。

- 为什么 alltoall 不需要分层（对比 allgather 的 ÷L 共享、allreduce 的归约压缩）：
  i→j 的 chunk 是 j 独有数据，无论怎么绕都必须恰好过 die 边界 1 次——
  **直拉本就踩在跨界流量下界上**，剩余优化只有并行（≥16KB 按 rank 扇出）
  与拷贝质量
- 纯拉模型：每人只写自己的 recvbuf；流式拷贝依赖 kupl_memcpy（SDMA 可用则 DMA）

## 基准（MPI 拉起）

控制变量法：每个场景内 mpirun 布局固定，唯一变量是算法。脚本都在 `examples/`，
结果写 `./logs/<时间戳>/`（不进仓库）。

| 脚本 | 矩阵 | 判定 |
|------|------|------|
| `bench_ag_mpi.sh` | allgather：3 布局 × linear/分层/auto（auto=拓扑路由，应与分层列一致） | 逐档 Δ%（相对 linear） |
| `bench_ar_mpi.sh` | allreduce：3 布局 × plain/分层/auto | SUM uint8 逐元素 VERIFY（Σrank mod 256） |
| `bench_a2a_mpi.sh` | alltoall：direct vs native | 来源/目的 rank + 下标校验，失败即停留日志 |
| `bench_ar_compare.sh` | 对比 6 列：kupl 原生/分层/auto + xucg 参照 + torch-kupl + torch-gloo | Δhier%（算法）+ Δprod%（vs gloo，目标 ≥30%）；kupl 列 VERIFY 失败即非零退出，xucg 仅警告 |

```bash
source ../kuccl_env.sh
cd examples
./bench_ag_mpi.sh 16
./bench_ar_mpi.sh 16
./bench_a2a_mpi.sh 16
./bench_ar_compare.sh 16           # REPEATS=3 STAT=min（抗抖动）
```

场景：same-die（全部 rank 一个 NUMA 节点）、cross-die（前 2 节点各半）、
cross-socket（2 socket 各半）。**分层收益依赖绑核布局**——same-die 应与
plain 持平（安全网），cross-die 看 die 粒度、cross-socket 看 socket 粒度；
不绑核跑基准没有意义。

## 日志与 TLS

- 日志：`KUCCL_LOG_LEVEL`（`OFF/ERROR/WARN/INFO/TRACE`，默认 `WARN`），输出 stderr。
- 落盘：`KUCCL_LOG_FILE=/tmp/kuccl.log`（stderr 之外同步写文件；路径含 `%r` 时按
  rank 分文件，多进程无交错——跑分时查关键日志用）。
- rank 过滤：`KUCCL_LOG_RANKS="0,3-5"`（只放行列出的 rank，大 world 防刷屏）。
- 耗时插桩：`KUCCL_KUPL_TIMING=1`（kupl 后端分阶段耗时汇总，comm 销毁时输出）。
- TLS：单机 `sm,self`；跨节点 `rc,sm,self`（节点内走共享内存，node-aware 算法依赖）。

## 支持矩阵与已知限制

### 支持矩阵

| 能力 | xucg（单机+跨节点） | kupl（单机 shm） |
|------|------|------|
| AllReduce SUM | ✅ 原生 | ✅ KUPL 原生 |
| AllReduce MIN/MAX | ✅ 原生 | ⚠️ 本地模拟（KUPL 底层仅 SUM），功能语义与原生一致 |
| AllGather / Broadcast | ✅ | ✅ |
| AllToAll（等长） | ✅ | ✅ |
| dtype | int8/uint8/int32/uint32/int64/uint64/float32/float64 | 同左（CHAR/INT/LONG/FLOAT/DOUBLE 映射，见 `kupl.cpp map_datatype`） |
| 变长 AllToAll / 点对点 | ❌ 不在公开面 | ❌ |

### 已知限制

1. **kupl 后端为同步驱动**：集合算子在 `coll_start` 内完成全部动作，
   request 首次 Test 即 COMPLETE；`KUCCL_TEST_TIMEOUT_MS` 看门狗保护不到
   算子内部。
2. **单请求字节数上限 2GB**（KUPL 侧 count 为 int32）。
