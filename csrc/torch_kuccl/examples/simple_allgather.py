#!/usr/bin/env python3
"""
simple_allgather.py —— 极简 allgather 示例

从旧文件 test_kuccl_py.py 简化而来，展示 KUCCL 的最简用法。

重构后只需 import kuccl 即自动注册 backend，无需手动 register_backend。

运行方式：
    python examples/simple_allgather.py
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# import kuccl 即自动注册 "kuccl" backend
import kuccl  # noqa: F401


def worker(rank: int, world_size: int):
    """单个 rank 的最小示例。"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29512"

    dist.init_process_group(backend="kuccl", rank=rank, world_size=world_size)
    print(f"[rank {rank}] process group 就绪", flush=True)

    # 各 rank 贡献 {rank*10, rank*10+1, ...}
    send = torch.arange(rank * 10, rank * 10 + 4, dtype=torch.int32)
    gather_list = [torch.zeros_like(send) for _ in range(world_size)]
    dist.all_gather(gather_list, send)
    print(f"[rank {rank}] all_gather = {[x.tolist() for x in gather_list]}", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] 完成", flush=True)


def main():
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
    print("ALL PASSED", flush=True)


if __name__ == "__main__":
    main()
