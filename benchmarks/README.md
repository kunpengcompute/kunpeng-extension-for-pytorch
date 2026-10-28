# KUCCL Benchmarks

> 基于 DeepSpeedExamples 移植的基准套件 + kuccl 对比工具。
> 正确性测试见 [communication/PYTORCH_CPU_TESTS.md](communication/PYTORCH_CPU_TESTS.md)。

## 目录结构

| 目录 | 内容 | 用途 |
|---|---|---|
| `communication/` | DeepSpeed 通信基准套件（已适配 kuccl） | 延迟 / BusBW，业界可对比 |
| `autosp/` | AutoSP 分布式注意力 benchmark | 端到端场景 |
| `deepcompile/` | DeepCompile benchmark | 端到端场景 |
| `inference/` | 推理 benchmark | 端到端场景 |

## 鲲鹏适配（communication/）

DeepSpeed 套件原生面向 GPU + deepspeed 启动器，四处改动让它跑在鲲鹏 CPU 上：

- `cpu_compat.py`：加速器垫片——装了 DeepSpeed 就转发 `get_accelerator`
  （`DS_ACCELERATOR=cpu`），没装给纯 CPU stub；GPU Event 换墙钟计时
- `run_cpu.py`：本地多进程拉起 + 绑核（NUMA die / socket 布局、一人一核、
  OMP=1），不经 deepspeed / mpi
- 消息规模 local patch（all_reduce / all_gather / broadcast / all_to_all
  各一处，代码里有标记）：每 rank 消息 M 与扫描档一致——原版 ws×M 的构造
  在 CPU 大 ws 下内存失控；all_to_all 的 M < world_size 跳过
- `compare_run.py`：零侵入双后端编排——不改套件文件，`--backend kuccl/gloo`
  同参各跑一遍，逐尺寸对齐算提升，平均 ≥30% 时退出码 0；BusBW 与业界同式

## 快速开始

先构建两包（见 `csrc/kuccl`、`csrc/torch_kuccl` README）。

```bash
cd benchmarks/communication

python run_cpu.py --world-size 2 -- all_reduce.py --dist torch \
    --backend kuccl --device cpu --scan --maxsize 19

# 双后端对比 + 判定（一条命令）
python compare_run.py --ops all_reduce --world-size 2 --maxsize 19
```

## Benchmark 原始说明（DeepSpeedExamples 上游）

All benchmarks that use the DeepSpeed library are maintained in this folder. We welcome contributions in this space!
