# KUCCL Benchmarks

> 基于 DeepSpeedExamples 移植的基准套件 + 自研验收工具。
> 所有命令的完整速查见 **[communication/bash.md](communication/bash.md)**。

## 目录结构

| 目录 | 内容 | 用途 |
|---|---|---|
| `communication/` | DeepSpeed 通信基准套件（已适配 kuccl 后端） | 第三方口径佐证（BusBW），业界可对比 |
| `autosp/` | AutoSP 分布式注意力 benchmark | 端到端场景 |
| `deepcompile/` | DeepCompile benchmark | 端到端场景 |
| `inference/` | 推理 benchmark | 端到端场景 |

## 快速开始（两条命令）

```bash
# 环境激活（每次开新 shell）
cd /root/pacific_ext/w30084706/test/kuccl
source ./env.sh
export PYTHONPATH=$PWD:$PYTHONPATH

# all_reduce：kuccl vs gloo（在 benchmarks/communication/ 下）
cd benchmarks/communication
python run_cpu.py --world-size 2 -- all_reduce.py --dist torch --backend kuccl --device cpu --scan --maxsize 19
python run_cpu.py --world-size 2 -- all_reduce.py --dist torch --backend gloo  --device cpu --scan --maxsize 19
```

## 两套性能工具的分工

| | `communication/`（本目录） | `../examples/benchmark.py` |
|---|---|---|
| 来源 | DeepSpeed 官方（业界可对比） | 自研（设计书验收口径） |
| 输出 | 延迟 + Throughput + BusBW | 延迟 + 带宽 + 自动算提升% + CSV |
| 后端对比 | 手动 `--backend` 各跑一遍 | 一条命令双后端自动对比 |
| 多节点 | torchrun 手动 | 内建 `--bind-nics` 网卡轮转 |
| 验收判定 | 无（人工对比） | 自动标 ★ 达标（对齐设计书 +30%） |

**结论**：验收数据出 `examples/benchmark.py`；本套件跑关键算子做第三方佐证。

## 相关文档

- `bash.md`（communication/ 内）—— 命令速查 + 参数变体 + 多节点 + 踩坑表
- `benchmark_kuccl.md`（本目录）—— kuccl 后端适配的分析与测试记录
- `../docs/` —— 编译踩坑、重构说明、迁移对照等

## Benchmark 原始说明（DeepSpeedExamples 上游）

All benchmarks that use the DeepSpeed library are maintained in this folder. We welcome contributions in this space!
