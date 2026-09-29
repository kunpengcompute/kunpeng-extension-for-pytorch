# kuccl — 鲲鹏集合通信库

基于 UCG 公有接口的纯 C 集合通信库（`libkuccl.so`），零框架依赖。
PyTorch 接入见 [torch_kuccl](../torch_kuccl/README.md)。

## 目录结构

```
kuccl/
├── include/          公开头文件（kuccl.h / types.hpp）
├── src/              实现（coll / comm / log / error + backend/{xucg,kupl}）
├── examples/         coll_test_mpi.cpp（MPI 直调用例）
├── tests/            coll_test.cpp（C 功能测试）
├── kuccl_env.sh      库环境（HPCKit / HUCX / XUCG 路径）
└── build.sh / CMakeLists.txt / kuccl.map
```

## 环境与构建

前置：HPCKit（含 HUCX / XUCG）。默认路径在 `kuccl_env.sh`，可外部覆盖：

```bash
source ./kuccl_env.sh          # 路径不对时：先 export HPCKIT=<实际安装路径>
./build.sh                     # 产物：build/libkuccl.so
```

## MPI 直调用例

```bash
source ./kuccl_env.sh
export PATH=$HPCKIT/hmpi/bisheng/release/hmpi/bin:$PATH
export LD_LIBRARY_PATH=$HPCKIT/hmpi/bisheng/release/hmpi/lib:$LD_LIBRARY_PATH

cd examples
mpicxx coll_test_mpi.cpp -I../include -L../build -lkuccl -o coll_test_mpi
mpirun --allow-run-as-root -np 4 --mca coll ^ucg ./coll_test_mpi
```

预期：AllReduce / Broadcast / AllGather / AllToAll 四项 `PASS`。

## 对外接口（kuccl.h）

| 类别 | 接口 |
|------|------|
| 通信域 | `kucclGetUniqueId` `kucclCommInitRankConfig` `kucclCommSplit` `kucclCommDestroy` `kucclCommAbort` |
| 集合算子 | `kucclAllReduce` `kucclAllGather` `kucclBroadcast` `kucclAllToAll` |
| 异步 / 辅助 | `kucclTest` `kucclGetLastError` `kucclFlushComm` |

AllToAllV 与点对点不在公开面。

## 后端

| 后端 | 状态 | 说明 |
|------|------|------|
| `xucg` | 默认 | UCG 公有接口，单机 + 跨节点 |
| `kupl` | 单机 | KUPL shm 快路径，需 `KUPL_DIR` 指向安装前缀重新编译 |

选择：环境变量 `KUCCL_BACKEND_TYPE` 或 `kucclConfig_t.backend`。

## 日志与 TLS

- 日志：`KUCCL_LOG_LEVEL`（`OFF/ERROR/WARN/INFO/TRACE`，默认 `WARN`），输出 stderr。
- TLS：单机 `sm,self`；跨节点 `rc,sm,self`（节点内走共享内存，node-aware 算法依赖）。
