#!/usr/bin/env python3
"""
test_allreduce.py —— AllReduce 集合通信测试

合并自旧文件 test_allreduce.py + test_allreduce2.py

测试内容：
  1. int32 MIN：验证"按元素"归约（非逐字节）——256 vs 255 区分字节级和元素级
  2. float32 SUM：经典求和
  3. int32 MAX：含负数

关键验证点：
  allreduce 必须用真实 dtype（不能用 UINT8），否则逐字节归约会出错。
  用例 1 的 256 vs 255 就是专门验证这一点的。

运行方式：
    python tests/test_allreduce.py
"""

import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import spawn_and_run, init_group, rank_log


def check(rank, got, expected, tag):
    """校验并打印结果。"""
    ok = torch.equal(got, expected)
    rank_log(
        rank,
        f"{tag}: got={got.tolist()} expected={expected.tolist()} "
        f"-> {'PASS' if ok else 'FAIL'}",
    )
    return ok


def worker(rank: int, world_size: int, port: int):
    """单个 rank 的测试逻辑。"""
    init_group(rank, world_size, port)
    all_ok = True

    # ── 用例 1：int32 MIN —— 验证"按元素"而非"按字节"归约 ──────
    #   rank0 [256,1000,-5,7]，rank1 [255,999,3,8]
    #   逐字节 min 会给错（0xFF 截断），元素级 min 才正确
    #   期望 min = [255, 999, -5, 7]
    y = torch.tensor(
        [256, 1000, -5, 7] if rank == 0 else [255, 999, 3, 8], dtype=torch.int32
    )
    dist.all_reduce(y, op=dist.ReduceOp.MIN)
    all_ok &= check(
        rank,
        y,
        torch.tensor([255, 999, -5, 7], dtype=torch.int32),
        "int32 min (byte-vs-element)",
    )

    # ── 用例 2：float32 SUM ──────────────────────────────────
    #   各 rank 贡献 [rank, rank, ...]，SUM = [0+1+..+ws-1, ...]
    s = torch.full((4,), float(rank), dtype=torch.float32)
    dist.all_reduce(s, op=dist.ReduceOp.SUM)
    expected_sum = float(sum(range(world_size)))
    all_ok &= check(
        rank, s, torch.full((4,), expected_sum, dtype=torch.float32), "float32 sum"
    )

    # ── 用例 3：int32 MAX（含负数）──────────────────────────
    m = torch.tensor([-rank, rank * 10, -100, rank], dtype=torch.int32)
    dist.all_reduce(m, op=dist.ReduceOp.MAX)
    expected_max = torch.tensor(
        [
            max(-r for r in range(world_size)),
            max(r * 10 for r in range(world_size)),
            -100,
            max(r for r in range(world_size)),
        ],
        dtype=torch.int32,
    )
    all_ok &= check(rank, m, expected_max, "int32 max")

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
