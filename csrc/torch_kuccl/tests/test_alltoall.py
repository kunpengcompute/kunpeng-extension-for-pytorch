#!/usr/bin/env python3
"""
test_alltoall.py —— AllToAll 集合通信测试

重构自旧文件 test_alltoall.py

测试内容：
  1. 等长 all_to_all_single：各 rank 等分发送缓冲
  2. 变长 all_to_all_single：带 outputSplitSizes / inputSplitSizes

运行方式：
    python tests/test_alltoall.py
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

    # ── 用例 1：等长 all_to_all_single（int32）──────────────
    #   每 rank 发送 world_size 块，每块 2 个元素
    #   rank 0: [0,1, 2,3]     → 第 0 块给 r0，第 1 块给 r1
    #   rank 1: [10,11, 12,13] → 第 0 块给 r0，第 1 块给 r1
    #   期望 r0 recv = [0,1, 10,11]，r1 recv = [2,3, 12,13]
    per_rank = 2
    send = torch.arange(rank * 10, rank * 10 + world_size * per_rank, dtype=torch.int32)
    recv = torch.empty(world_size * per_rank, dtype=torch.int32)
    dist.all_to_all_single(recv, send)

    # 构造期望值：recv[j] = rank j 的第 rank 块
    expected = torch.empty(world_size * per_rank, dtype=torch.int32)
    for src in range(world_size):
        # src rank 的第 rank 块
        block = torch.arange(src * 10 + rank * per_rank,
                             src * 10 + rank * per_rank + per_rank,
                             dtype=torch.int32)
        expected[src * per_rank : (src + 1) * per_rank] = block

    ok = torch.equal(recv, expected)
    rank_log(rank, f"等长 alltoall: {'PASS' if ok else 'FAIL'} "
             f"(recv={recv.tolist()})")
    all_ok &= ok

    # ── 用例 2：变长 all_to_all_single（数值校验，任意 world_size≥2）──
    #   发送矩阵 M[r][j] = (r+j) % 3（对称：r 发给 j 的量 == j 从 r 收的量，
    #   自环对 M[r][r] 天然一致，不存在配对越界）
    #   用 float32（4 字节元素）：若底层把元素错当字节，数值必错
    if world_size >= 2:
        M = [[(r + j) % 3 for j in range(world_size)]
             for r in range(world_size)]
        in_splits = M[rank]                                # 发给 j 的元素数
        out_splits = [M[j][rank] for j in range(world_size)]  # 从 j 收的元素数
        in_total = sum(in_splits)
        out_total = sum(out_splits)

        send_v = (rank * 1000.0
                  + torch.arange(in_total, dtype=torch.float32))
        recv_v = torch.empty(out_total, dtype=torch.float32)
        dist.all_to_all_single(recv_v, send_v,
                               output_split_sizes=out_splits,
                               input_split_sizes=in_splits)

        # 期望：recv 第 j 段 = rank j 的发送缓冲中发给本 rank 的那一段
        expected = torch.empty(out_total, dtype=torch.float32)
        pos = 0
        for j in range(world_size):
            off = sum(M[j][k] for k in range(rank))  # j 缓冲中位于本段之前的量
            n = M[j][rank]
            expected[pos:pos + n] = (
                j * 1000.0
                + torch.arange(off, off + n, dtype=torch.float32))
            pos += n

        ok = torch.equal(recv_v, expected)
        rank_log(rank, f"变长 alltoall(数值): {'PASS' if ok else 'FAIL'} "
                 f"(recv={recv_v.tolist()} expected={expected.tolist()})")
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
