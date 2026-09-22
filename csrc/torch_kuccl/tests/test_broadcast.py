#!/usr/bin/env python3
"""
test_broadcast.py —— Broadcast 集合通信测试

重构自旧文件 test_broadcast.py + test_broadcast2.py

修复的 bug：
  旧 test_broadcast.py 硬编码 spawn 256 进程，大多数机器跑不动。
  改为可配置 world_size，默认 2 进程。

测试内容：
  1. float32 广播：root 贡献数据，其他 rank 验证一致性
  2. 大数据量广播：1024 个元素，模拟模型参数分发

运行方式：
    python tests/test_broadcast.py
    WORLD_SIZE=4 python tests/test_broadcast.py
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

    # ── 用例 1：float32 小数据广播 ──────────────────────────
    num_elements = 8
    root = 0
    if rank == root:
        tensor = torch.arange(num_elements, dtype=torch.float32)
    else:
        tensor = torch.zeros(num_elements, dtype=torch.float32)

    rank_log(rank, f"广播前: {tensor[:4].tolist()} ...")
    dist.broadcast(tensor, src=root)

    expected = torch.arange(num_elements, dtype=torch.float32)
    ok = torch.equal(tensor, expected)
    rank_log(rank, f"广播后: {tensor[:4].tolist()} ... -> {'PASS' if ok else 'FAIL'}")
    all_ok &= ok

    # ── 用例 2：大数据量广播（模拟参数分发）─────────────────
    large_size = 1024
    if rank == root:
        big = torch.randn(large_size, dtype=torch.float32)
    else:
        big = torch.zeros(large_size, dtype=torch.float32)

    dist.broadcast(big, src=root)

    # 所有 rank 的 big 应与 root 的原始数据一致
    # 通过 allreduce 验证所有 rank 数据相同（sum 后每元素 = 原值 * world_size）
    check_sum = big.clone()
    dist.all_reduce(check_sum, op=dist.ReduceOp.SUM)
    expected_sum = big * world_size
    ok_big = torch.equal(check_sum, expected_sum)
    rank_log(rank, f"大数据广播: {'PASS' if ok_big else 'FAIL'}")
    all_ok &= ok_big

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
