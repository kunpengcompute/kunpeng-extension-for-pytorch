#!/usr/bin/env python3
"""
test_allgather.py —— AllGather 集合通信测试

合并自旧文件 test_allgather.py + test_kuccl_py.py

测试内容：
  1. 基础 allgather：各 rank 贡献不同数据，验证拼接结果正确
  2. all_gather_into_tensor：单块连续缓冲的 allgather
  3. int32 / float32 多 dtype 覆盖

运行方式：
    python tests/test_allgather.py
    # 或 pytest：
    pytest tests/test_allgather.py -v
"""

import os
import sys
import torch
import torch.distributed as dist

# 复用 conftest 的启动工具
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import spawn_and_run, init_group, rank_log


def worker(rank: int, world_size: int, port: int):
    """单个 rank 的测试逻辑。"""
    init_group(rank, world_size, port)
    all_ok = True

    # ── 用例 1：基础 allgather（int32）──────────────────────
    #   rank 0 贡献 [0,1,2,3]，rank 1 贡献 [10,11,12,13]
    #   期望双方 recv = [0,1,2,3, 10,11,12,13]
    count = 4
    send = torch.arange(rank * 10, rank * 10 + count, dtype=torch.int32)
    gather_list = [torch.zeros(count, dtype=torch.int32) for _ in range(world_size)]
    dist.all_gather(gather_list, send)

    expected = torch.tensor(
        [r * 10 + i for r in range(world_size) for i in range(count)],
        dtype=torch.int32,
    )
    recv_flat = torch.cat(gather_list)
    ok = torch.equal(recv_flat, expected)
    rank_log(rank, f"allgather int32: {'PASS' if ok else 'FAIL'} "
             f"(recv={recv_flat.tolist()})")
    all_ok &= ok

    # ── 用例 2：all_gather_into_tensor（float32）────────────
    #   单块连续缓冲，比列表式更高效
    send_f = torch.ones(8, dtype=torch.float32) * rank
    recv_f = torch.empty(world_size * 8, dtype=torch.float32)
    dist.all_gather_into_tensor(recv_f, send_f)

    expected_f = torch.cat([torch.ones(8, dtype=torch.float32) * r for r in range(world_size)])
    ok_f = torch.equal(recv_f, expected_f)
    rank_log(rank, f"all_gather_into_tensor float32: {'PASS' if ok_f else 'FAIL'}")
    all_ok &= ok_f

    dist.barrier()
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
