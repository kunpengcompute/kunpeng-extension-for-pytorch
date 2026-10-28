#!/usr/bin/env python3
"""
conftest.py —— pytest 共享 fixtures

提供分布式测试的进程级启动与 TCPStore 初始化。
所有 test_*.py 通过 fixture 获取 rank / world_size / store，
避免每个测试文件各自写一套 spawn 模板。

使用方式：
    pytest tests/ -v
    # 或手动运行单个文件：
    python tests/test_allgather.py
"""

import os
from datetime import timedelta
from typing import Callable

import torch.distributed as dist

# import kuccl 即自动注册 "kuccl" backend（无需手动 register_backend）
import kuccl  # noqa: F401


# ============================================================================
# 默认参数
# ============================================================================

DEFAULT_WORLD_SIZE = 2
DEFAULT_MASTER_ADDR = "127.0.0.1"
DEFAULT_MASTER_PORT = 29500


def spawn_and_run(
    worker_fn: Callable,
    world_size: int = DEFAULT_WORLD_SIZE,
    master_port: int = DEFAULT_MASTER_PORT,
) -> int:
    """
    用 mp.spawn 启动 N 个子进程，每个进程调用 worker_fn(rank, world_size, port)。

    返回 0 表示全部 PASS，非 0 表示有失败。

    为什么用 spawn 而非 fork：
        UCX 在 fork 后的子进程中会出现 fd 错乱（fork-after-init 坑）。
        spawn 是重新 exec 一个干净进程，UCX 在子进程内从头初始化。
    """
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(target=worker_fn, args=(r, world_size, master_port))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    rc = 0
    for p in procs:
        p.join()
        rc |= p.exitcode or 0
    return rc


def init_group(rank: int, world_size: int, port: int, timeout_sec: int = 60):
    """初始化 process group，返回后即可使用 dist.all_gather 等 API。"""
    os.environ["MASTER_ADDR"] = DEFAULT_MASTER_ADDR
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend="kuccl",
        init_method=f"tcp://{DEFAULT_MASTER_ADDR}:{port}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=timeout_sec),
    )


def rank_log(rank: int, msg: str) -> None:
    """带 rank 前缀的日志输出。"""
    print(f"[Rank {rank}] {msg}", flush=True)
