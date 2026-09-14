#!/usr/bin/env python3
"""
test_spec_case1.py —— 设计书 2.4.5 验收用例 1：PyTorch 加载路径

验收口径（设计书示例："cpu:kuccl"，
即设备:后端 格式；本测试按正确口径实现）：
    import kuccl
    dist.init_process_group("cpu:kuccl", rank=..., world_size=...)
    dist.all_reduce / dist.broadcast / dist.all_gather / dist.all_to_all

校验内容（4 个必测算子全部走 kuccl，逐一数值校验）：
    all_reduce   float32 SUM
    broadcast    src=0
    all_gather   列表式
    all_to_all   all_to_all_single

运行：
    python tests/test_spec_case1.py                  # 默认 world_size=2
    WORLD_SIZE=4 python tests/test_spec_case1.py
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import spawn_and_run, rank_log


def worker(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend="cpu:kuccl",           # 设计书正确口径（设备:后端 格式）
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=world_size,
    )
    rank_log(rank, "init_process_group(backend='cpu:kuccl') OK")
    all_ok = True

    # ── 1. all_reduce：各 rank 填 rank，期望 sum(range(ws)) ──
    y = torch.full((4,), float(rank), dtype=torch.float32)
    dist.all_reduce(y)
    ok = torch.equal(y, torch.full((4,), float(sum(range(world_size)))))
    rank_log(rank, f"all_reduce: {'PASS' if ok else 'FAIL'} "
             f"(got={y.tolist()})")
    all_ok &= ok

    # ── 2. broadcast：src=0 填 42，期望全员 42 ──
    z = torch.full((4,), float(42 + rank), dtype=torch.float32)
    dist.broadcast(z, src=0)
    ok = torch.equal(z, torch.full((4,), 42.0))
    rank_log(rank, f"broadcast: {'PASS' if ok else 'FAIL'} "
             f"(got={z.tolist()})")
    all_ok &= ok

    # ── 3. all_gather：rank r 贡献 [r, r]，期望按 rank 顺序拼接 ──
    g = torch.full((2,), float(rank), dtype=torch.float32)
    out = [torch.empty(2, dtype=torch.float32) for _ in range(world_size)]
    dist.all_gather(out, g)
    ok = all(torch.equal(out[r], torch.full((2,), float(r)))
             for r in range(world_size))
    rank_log(rank, f"all_gather: {'PASS' if ok else 'FAIL'}")
    all_ok &= ok

    # ── 4. all_to_all：rank r 的第 j 块填 r*100+j，期望第 j 块收到 j*100+r ──
    per = 2
    s = torch.arange(rank * 100, rank * 100 + world_size * per,
                     dtype=torch.float32)
    r_ = torch.empty(world_size * per, dtype=torch.float32)
    dist.all_to_all_single(r_, s)
    exp = torch.empty(world_size * per, dtype=torch.float32)
    for j in range(world_size):
        exp[j * per:(j + 1) * per] = torch.arange(
            j * 100 + rank, j * 100 + rank + per, dtype=torch.float32)
    ok = torch.equal(r_, exp)
    rank_log(rank, f"all_to_all: {'PASS' if ok else 'FAIL'}")
    all_ok &= ok

    dist.barrier()
    rank_log(rank, f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    dist.destroy_process_group()

    if not all_ok:
        sys.exit(1)


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    rc = spawn_and_run(worker, world_size=world_size)
    print(f"{'ALL PASSED' if rc == 0 else 'SOME FAILED'} (exit={rc})", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
