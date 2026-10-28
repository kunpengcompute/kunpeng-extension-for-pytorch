#!/usr/bin/env python3
"""
test_init_timing.py —— init_process_group 耗时分解测试

重构自旧文件 test_kuccl_init_timing.py

测试内容：
  1. Python 端 perf_counter 包裹 init_process_group 测端到端
  2. C++ 端 KuCCL_TIMING=1 写出分段 JSON（需手动开环境变量）
  3. 做一次 allgather 确认通信正常

输出文件：
  kuccl_py_timing_rank<N>.json  —— Python 端端到端
  kuccl_timing_rank<N>.json     —— C++ 端 11 段（需 KuCCL_TIMING=1）

运行方式：
    # 不开 C++ 计时（仅测端到端）
    python tests/test_init_timing.py

    # 开 C++ 分段计时
    KuCCL_TIMING=1 python tests/test_init_timing.py

    # 然后画图：
    python plot_kuccl_timing.py --rank 0
"""

import json
import os
import sys
import time
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import spawn_and_run, init_group, rank_log


def worker(rank: int, world_size: int, port: int):
    """单个 rank 的测试逻辑。"""
    # ── 端到端计时起点 ──────────────────────────────────────
    t_start = time.perf_counter()

    init_group(rank, world_size, port)

    # ── 端到端计时终点 ──────────────────────────────────────
    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000.0

    rank_log(rank, f"init_process_group 耗时: {total_ms:.2f} ms")

    # ── 写出 Python 端计时 JSON ────────────────────────────
    py_timing = {
        "total_ms": round(total_ms, 3),
        "phases": [
            {"name": "init_process_group (端到端)", "ms": round(total_ms, 3)},
        ],
    }
    py_path = f"kuccl_py_timing_rank{rank}.json"
    with open(py_path, "w") as f:
        json.dump(py_timing, f, indent=2, ensure_ascii=False)
    rank_log(rank, f"Python 端计时已写入 {py_path}")

    # ── 做一次最小 allgather 确认通信正常 ───────────────────
    t = torch.tensor([rank], dtype=torch.int32)
    out = [torch.zeros(1, dtype=torch.int32) for _ in range(world_size)]
    dist.all_gather(out, t)
    got = [x.item() for x in out]
    ok = got == list(range(world_size))
    rank_log(rank, f"allgather 验证: {got} -> {'PASS' if ok else 'FAIL'}")

    dist.barrier()
    dist.destroy_process_group()

    if not ok:
        sys.exit(1)


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    rc = spawn_and_run(worker, world_size=world_size)
    print(f"{'ALL PASSED' if rc == 0 else 'SOME FAILED'} (exit={rc})", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
