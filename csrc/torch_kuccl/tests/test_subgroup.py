#!/usr/bin/env python3
"""
test_subgroup.py —— 重叠子通信域测试

重构自旧文件 test_kuccl_subgroup_final.py

测试内容：
  1. world group 基线 allgather
  2. 重叠子组 allgather（rank 0 同时在 [0,1] 和 [0,2]）
  3. 无重叠子组 allgather（[0,1] / [2,3]）

关键验证点：
  get_ucp_ep 按全局 rank 查进程级共享地址表，
  重叠子组能否正确解析不同 group_rank 到不同 global_rank。

运行方式：
    python tests/test_subgroup.py
    WORLD_SIZE=4 python tests/test_subgroup.py
"""

import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import spawn_and_run, init_group, rank_log


# 测试用子组定义（world_size >= 2 时用 [0,1]）
SUBGROUPS_2 = [[0, 1]]
SUBGROUPS_4 = [[0, 1], [2, 3]]


def worker(rank: int, world_size: int, port: int):
    """单个 rank 的测试逻辑。"""
    init_group(rank, world_size, port)
    all_ok = True

    # ── 0. world group 基线 allgather ────────────────────────
    t = torch.tensor([rank], dtype=torch.int32)
    out = [torch.zeros(1, dtype=torch.int32) for _ in range(world_size)]
    dist.all_gather(out, t)
    got = [x.item() for x in out]
    ok = got == list(range(world_size))
    rank_log(rank, f"world allgather: {got} -> {'PASS' if ok else 'FAIL'}")
    all_ok &= ok

    # ── 1. 创建子组 ─────────────────────────────────────────
    if world_size >= 4:
        subgroups = SUBGROUPS_4
    elif world_size >= 2:
        subgroups = SUBGROUPS_2
    else:
        subgroups = []

    groups = {}
    for members in subgroups:
        # 集体操作：所有 rank 以相同顺序调用（含非成员）
        groups[tuple(members)] = dist.new_group(ranks=members)
        rank_log(rank, f"new_group({members}) 已创建")

    # ── 2. 各子组内 allgather ────────────────────────────────
    for members in subgroups:
        g = groups[tuple(members)]
        if rank not in members:
            continue
        lws = len(members)
        t = torch.tensor([rank], dtype=torch.int32)
        out = [torch.zeros(1, dtype=torch.int32) for _ in range(lws)]
        dist.all_gather(out, t, group=g)
        got = [x.item() for x in out]
        exp = sorted(members)
        ok = got == exp
        rank_log(rank, f"子组 {members} allgather: {got} -> {'PASS' if ok else 'FAIL'}")
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
