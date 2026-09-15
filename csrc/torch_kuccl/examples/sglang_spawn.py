#!/usr/bin/env python3
"""
sglang_spawn.py —— 模拟 SGLang 多进程启动

重构自旧文件 test_kuccl_sglang.py

验证 KUCCL 后端能在 mp.spawn 出的子进程中正常工作
（避免 fork-after-init 的 fd 错乱坑）。

运行方式：
    python examples/sglang_spawn.py --tp-size 2
"""

import argparse
import os
import torch
import torch.multiprocessing as mp

import kuccl  # noqa: F401 — 自动注册 backend


def worker(rank: int, world_size: int, master_addr: str, master_port: int):
    """模拟 SGLang 的 TP 子进程。"""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    print(f"[Rank {rank}] 子进程启动，初始化 process group...", flush=True)

    import torch.distributed as dist
    dist.init_process_group(
        backend="kuccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )

    print(f"[Rank {rank}] process group 就绪", flush=True)

    # 模拟 SGLang 做一次 allgather
    tensor = torch.tensor([rank * 10 + i for i in range(4)], dtype=torch.int32)
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)

    print(f"[Rank {rank}] all_gather 结果: {[t.tolist() for t in gather_list]}",
          flush=True)

    dist.destroy_process_group()
    print(f"[Rank {rank}] 完成", flush=True)


def main():
    parser = argparse.ArgumentParser(description="模拟 SGLang spawn 启动")
    parser.add_argument("--tp-size", type=int, default=2,
                        help="张量并行大小（进程数）")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=23457)
    args = parser.parse_args()

    print(f"启动 {args.tp_size} 个子进程...", flush=True)
    mp.spawn(
        worker,
        args=(args.tp_size, args.master_addr, args.master_port),
        nprocs=args.tp_size,
        join=True,
    )
    print("所有进程完成", flush=True)


if __name__ == "__main__":
    main()
