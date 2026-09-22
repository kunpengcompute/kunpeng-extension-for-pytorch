# kuccl — 鲲鹏集合通信库

鲲鹏平台纯 C 集合通信库（`libkuccl.so`），基于 UCG（xUCG）公有接口实现，
零框架依赖——不依赖 PyTorch / MPI / Python。

> PyTorch 接入请使用兄弟目录的 [torch_kuccl](../torch_kuccl/README.md)（适配层）。

## 目录结构

```
kuccl/
├── include/           公开头文件（kuccl.h / types.hpp）
├── src/               实现（骨架层 + backend/ 双后端）
│   ├── coll.cpp       集合算子骨架（校验/锁/分发/看门狗）
│   ├── comm.cpp       通信域管理骨架（GetUniqueId/Init/Split/Destroy/Abort）
│   ├── log.{hpp,cpp}  统一日志（KUCCL_TRACE/INFO/WARN/ERROR）
│   ├── error.cpp      kucclGetLastError（TLS 错误环）
│   └── backend/
│       ├── backend.hpp/cpp   后端接口 + 注册表
│       ├── xucg.cpp          xucg 后端（UCG 公有接口，默认）
│       └── kupl.cpp          kupl 后端（预留存根）
├── examples/coll_test_mpi.cpp   MPI 直调完整示例（可编译可直接跑）
├── tests/coll_test.cpp          C 层功能测试
├── CMakeLists.txt / kuccl.map / build.sh / env.sh
```

## 双后端设计（设计书 2.4.3）

| 后端 | 状态 | 定位 |
|------|------|------|
| `xucg` | 可用（默认） | UCG 公有接口集合通信：单机 + 跨节点 |
| `kupl` | 可用（单机） | KUPL shm 共享内存快路径（`KUCCL_BACKEND_TYPE=kupl`，跨节点不支持） |

kupl 后端能力：AllReduce / Broadcast / AllToAll / AllGather（等长 alltoall 转换）。
需设置 `KUPL_DIR` 指向 kupl 安装前缀重新编译（未链接时该后端报"未构建"错误）。

选择方式：`kucclConfig_t.backend = "xucg"`（或环境变量 `KUCCL_BACKEND_TYPE`）。
后端边界：集合请求的状态判定与推进只走 UCG 公有接口
（`ucg_request_*_init / ucg_request_start / ucg_request_test / ucg_progress`，
状态码 `UCG_OK / UCG_INPROGRESS`）；UCP 仅作为 xucg 后端私有资源层
（ucg 编程模型要求注入 ucp 资源），封闭在 `backend/xucg.cpp` 内。

## 对外接口（12 个公开接口，kuccl.h）

| 类别 | 接口 |
|------|------|
| 通信域管理 | `kucclGetUniqueId` `kucclCommInitRankConfig` `kucclCommDestroy` `kucclCommSplit` |
| 集合通信（4 个） | `kucclAllGather` `kucclAllReduce` `kucclBroadcast` `kucclAllToAll` |
| 异步测试 | `kucclTest` |
| 生产化设施 | `kucclGetLastError` `kucclCommAbort` `kucclFlushComm` |

变长全交换（AllToAllV）与点对点 Send/Recv 不在公开面（等长 AllToAll 已覆盖
主力场景；变长/点对点需要时经 gloo/MPI）。

## 快速开始

```bash
# 0) 环境（HUCX/XUCG/conda 一键检查）
source env.sh --check

# 1) 构建库本体（本目录；适配层在 ../torch_kuccl 下独立构建）
./build.sh

# 2) MPI 直调（完整可编译示例，构建/运行命令见文件头注释）
cd examples && mpicxx coll_test_mpi.cpp -lkuccl ...   # 详见 coll_test_mpi.cpp

# 3) PyTorch 正确性/性能（在 torch_kuccl/ 下）
cd ../torch_kuccl
python tests/correctness.py --world-size 2
python examples/benchmark.py --csv r.csv
```

## C API 直调（MPI 程序）

完整可编译示例：[`examples/coll_test_mpi.cpp`](examples/coll_test_mpi.cpp)
（设计书 2.4.5 验收用例 2：MPI 进程直接拉起 kuccl，含建组/算子/销毁全流程与运行命令）。

## 日志

统一走 `src/log.{hpp,cpp}`，级别由 `KUCCL_LOG_LEVEL` 控制
（`OFF/ERROR/WARN/INFO/TRACE`，默认 `WARN` 生产静默），输出到 stderr，
格式：`[KUCCL][LEVEL][时间][rank N] message (file:line)`。

## 传输层配置（TLS 策略）

| 场景 | UCX_TLS | 说明 |
|------|---------|------|
| 单机 | `sm,self` | 共享内存快路径（前置：hucx 需启用 sm 组件） |
| 跨节点 | `rc,self` | rc（RoCE），不含 sm |
| 跨节点（备选） | `ud,self` | ud 替代 rc：大规模 EP 下更省资源 |
