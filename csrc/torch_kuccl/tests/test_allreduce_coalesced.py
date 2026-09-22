#!/usr/bin/env python3
"""
test_allreduce_coalesced.py —— 多 tensor 合并归约测试

验证 KuCCL 的 allreduce_coalesced 实现：
  把 N 个 tensor 拼成 1 个大 buffer，只做 1 轮通信。

对比：
  - 单 tensor allreduce（快速路径，不拼 buffer）
  - 多 tensor allreduce（coalesced 路径，拼 buffer）
  - 两种方式结果应一致

运行方式：
    python tests/test_allreduce_coalesced.py
"""

import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import spawn_and_run, init_group, rank_log


def worker(rank: int, world_size: int, port: int):
    """单个 rank 的测试逻辑。"""
    init_group(rank, world_size, port)
    all_ok = True

    # ── 用例 1：单 tensor（快速路径）─────────────────────────
    t1 = torch.tensor([rank * 10 + i for i in range(4)], dtype=torch.int32)
    expected1 = torch.tensor(
        [sum(r * 10 + i for r in range(world_size)) for i in range(4)],
        dtype=torch.int32,
    )
    dist.all_reduce(t1, op=dist.ReduceOp.SUM)
    ok1 = torch.equal(t1, expected1)
    rank_log(rank, f"单 tensor allreduce: {'PASS' if ok1 else 'FAIL'}")
    all_ok &= ok1

    # ── 用例 2：多 tensor（coalesced 路径）──────────────────
    #   3 个 tensor，各 4 个元素
    t2a = torch.full((4,), float(rank), dtype=torch.float32)
    t2b = torch.full((4,), float(rank * 2), dtype=torch.float32)
    t2c = torch.full((4,), float(rank * 3), dtype=torch.float32)

    expected_sum = float(sum(range(world_size)))
    expected2a = torch.full((4,), expected_sum, dtype=torch.float32)
    expected2b = torch.full((4,), expected_sum * 2, dtype=torch.float32)
    expected2c = torch.full((4,), expected_sum * 3, dtype=torch.float32)

    # allreduce_coalesced 会把 3 个 tensor 拼成 1 个 buffer，只做 1 轮通信
    dist.all_reduce_coalesced([t2a, t2b, t2c], op=dist.ReduceOp.SUM)

    ok2a = torch.equal(t2a, expected2a)
    ok2b = torch.equal(t2b, expected2b)
    ok2c = torch.equal(t2c, expected2c)
    rank_log(rank, f"多 tensor coalesced: a={'PASS' if ok2a else 'FAIL'} "
             f"b={'PASS' if ok2b else 'FAIL'} c={'PASS' if ok2c else 'FAIL'}")
    all_ok &= ok2a and ok2b and ok2c

    # ── 用例 3：多 tensor MIN（验证 dtype 没被错误处理）──────
    t3a = torch.tensor([rank, rank + 10], dtype=torch.int32)
    t3b = torch.tensor([rank * 100, rank * 100 + 1], dtype=torch.int32)

    dist.all_reduce_coalesced([t3a, t3b], op=dist.ReduceOp.MIN)

    expected3a = torch.tensor([0, 0], dtype=torch.int32)   # min(0,1,...,ws-1)
    expected3b = torch.tensor([0, 0], dtype=torch.int32)
    ok3 = torch.equal(t3a, expected3a) and torch.equal(t3b, expected3b)
    rank_log(rank, f"多 tensor MIN: {'PASS' if ok3 else 'FAIL'}")
    all_ok &= ok3

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
