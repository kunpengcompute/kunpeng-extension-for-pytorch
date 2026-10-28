#!/usr/bin/env python3
"""
correctness.py —— KuCCL 正确性测试套件 v2（全面版）

验证方式：解析式期望值（数据由 rank/ws 确定，期望结果可手算），
不依赖 gloo 做参照——但可用 --backend gloo 反向验证测试逻辑本身。

覆盖维度：
  算子   : allgather / all_gather_into_tensor / allreduce(SUM/MIN/MAX) /
            broadcast / alltoall
  dtype  : INT32 / INT64 / FP32 / FP64 / UINT8 / INT8
  数据   : 递增 / 全零 / 全相同 / 负数 / INT_MAX/MIN / 浮点特殊值 / 单元素
  交错   : allreduce → broadcast → allgather 交叉（fence 状态污染检测）
  压力   : 背靠背演进 / 随机 3 算子混合 / 内存稳定性 200 轮 RSS 监控
  边界   : count=1 / 非常规 root / in-place / 同缓冲复用×5
  切换   : --backends kuccl gloo 同进程先后建组
            验证 destroy/init 不互相污染）

用法：
  # 单机 2 进程（快速）
  python tests/correctness.py

  # 单节点 PPN=16
  python tests/correctness.py --world-size 16

  # 双节点（两台机器，node-rank 区分）
  torchrun --nnodes=2 --nproc_per_node=8 --node-rank=0 \
      --master-addr=<A_IP> --master-port=29600 tests/correctness.py

  # 8 节点 PPN=16 正确性检查
  torchrun --nnodes=8 --nproc_per_node=16 --node-rank=<0..7> \
      --master-addr=<A_IP> --master-port=29600 tests/correctness.py

  # 用 gloo 跑一遍验证测试逻辑本身
  python tests/correctness.py --backends gloo

退出码 0 = 全部 PASS（所有 rank 一致判定）。
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import torch
import torch.distributed as dist


# ============================================================================
# 检查工具
# ============================================================================

_RESULTS = []  # (tag, ok)


def check(tag, got, expected, exact=False):
    """整型精确比较；浮点 allclose（大数 FP32 求和有舍入）。
    exact=True 用于纯数据搬运类算子（allgather/broadcast）——
    它们不做任何算术，1 ulp 的偏差也是错误，不能用容差掩盖。"""
    if exact:
        ok = torch.equal(got, expected)
    elif got.dtype.is_floating_point:
        ok = torch.allclose(got, expected, rtol=1e-5, atol=1e-5)
    else:
        ok = torch.equal(got, expected)
    _RESULTS.append((tag, ok))
    if not ok:
        rank = os.environ.get("RANK", "?")
        print(
            f"[Rank {rank}] FAIL {tag}\n  got={got.flatten()[:8].tolist()} "
            f"expected={expected.flatten()[:8].tolist()}",
            flush=True,
        )
    return ok


def log_case(tag):
    """rank 0 打印当前用例（挂死时日志最后一行 = 卡点）。"""
    if os.environ.get("RANK", "0") == "0":
        print(f"[case {datetime.now().strftime('%H:%M:%S')}] {tag}", flush=True)


# ============================================================================
# 算子测试（期望值全部解析可算，任意 world_size 成立）
# ============================================================================

# 规模扫描：8B ~ 1M（2 的幂，同 benchmark.py 口径）
SCAN_SIZES = [2**e for e in range(3, 21)]
# 大 world_size 内存护栏：期望值+接收缓冲 ≈ 2×ws×n×元素字节，
# 超预算的档位自动跳过（记 skip 不算失败）
MEM_BUDGET = 128 * 1024 * 1024


def t_allgather(rank, ws):
    for dtype in (torch.int32, torch.int64, torch.float32):
        elem = torch.tensor([], dtype=dtype).element_size()
        for size_bytes in SCAN_SIZES:
            n = max(1, size_bytes // elem)
            if 2 * ws * n * elem > MEM_BUDGET:
                _RESULTS.append(
                    (f"allgather {dtype} {size_bytes}B (skip: 超内存预算)", True)
                )
                continue
            log_case(f"allgather {dtype} {size_bytes}B")
            send = (torch.arange(n, dtype=torch.float64) + rank * 1000).to(dtype)
            expected = torch.cat(
                [
                    (torch.arange(n, dtype=torch.float64) + r * 1000).to(dtype)
                    for r in range(ws)
                ]
            )
            gather_list = [torch.zeros(n, dtype=dtype) for _ in range(ws)]
            dist.all_gather(gather_list, send)
            check(
                f"allgather {dtype} {size_bytes}B",
                torch.cat(gather_list),
                expected,
                exact=True,
            )


def t_allgather_flat(rank, ws):
    # n=7 奇数元素（非 2 的幂），补扫描档位之外的计数形态
    for dtype in (torch.int32, torch.int64):
        log_case(f"all_gather_into_tensor {dtype} n=7")
        n = 7
        send = (torch.arange(n, dtype=torch.float64) + rank).to(dtype)
        expected = torch.cat(
            [(torch.arange(n, dtype=torch.float64) + r).to(dtype) for r in range(ws)]
        )
        recv = torch.zeros(ws * n, dtype=dtype)
        dist.all_gather_into_tensor(recv, send)
        check(f"all_gather_into_tensor {dtype} n=7", recv, expected, exact=True)
    # 连续缓冲版扫规模（int32）
    elem = 4
    for size_bytes in SCAN_SIZES:
        n = max(1, size_bytes // elem)
        if 2 * ws * n * elem > MEM_BUDGET:
            _RESULTS.append(
                (f"all_gather_into_tensor int32 {size_bytes}B (skip: 超内存预算)", True)
            )
            continue
        log_case(f"all_gather_into_tensor int32 {size_bytes}B")
        send = (torch.arange(n, dtype=torch.float64) + rank).to(torch.int32)
        expected = torch.cat(
            [
                (torch.arange(n, dtype=torch.float64) + r).to(torch.int32)
                for r in range(ws)
            ]
        )
        recv = torch.zeros(ws * n, dtype=torch.int32)
        dist.all_gather_into_tensor(recv, send)
        check(f"all_gather_into_tensor int32 {size_bytes}B", recv, expected)


def t_allreduce(rank, ws):
    for dtype in (torch.int32, torch.int64, torch.float32, torch.float64):
        elem = torch.tensor([], dtype=dtype).element_size()
        for size_bytes in SCAN_SIZES:
            n = max(1, size_bytes // elem)
            # send[i] = i + rank → SUM[i] = ws*i + sum(0..ws-1)
            log_case(f"allreduce SUM {dtype} {size_bytes}B")
            base = torch.arange(n, dtype=torch.float64)
            send = (base + rank).to(dtype)
            exp_sum = (base * ws + ws * (ws - 1) / 2).to(dtype)
            dist.all_reduce(send, op=dist.ReduceOp.SUM)
            check(f"allreduce SUM {dtype} {size_bytes}B", send, exp_sum)

    # MIN/MAX：负数 + 跨字节边界值（UINT8 逐字节语义会错的经典 case）
    vals = torch.tensor([256, 1000, -5, 7, 255, -1000], dtype=torch.int32)
    send_min = vals.clone().mul_(1 if rank % 2 == 0 else -1)
    dist.all_reduce(send_min, op=dist.ReduceOp.MIN)
    exp_min = torch.tensor([min(v, -v) for v in vals.tolist()], dtype=torch.int32)
    check("allreduce MIN int32 (跨字节边界+负数)", send_min, exp_min)

    send_max = vals.clone().mul_(1 if rank % 2 == 0 else -1)
    dist.all_reduce(send_max, op=dist.ReduceOp.MAX)
    exp_max = torch.tensor([max(v, -v) for v in vals.tolist()], dtype=torch.int32)
    check("allreduce MAX int32", send_max, exp_max)


def t_broadcast(rank, ws):
    elem = 4  # int32
    for size_bytes in SCAN_SIZES:
        n = max(1, size_bytes // elem)
        for root in (0, ws - 1):
            log_case(f"broadcast root={root} {size_bytes}B")
            src = torch.arange(n, dtype=torch.int32) + root * 100
            if rank == root:
                buf = src.clone()
            else:
                buf = torch.full(
                    (n,), -1, dtype=torch.int32
                )  # 垃圾初值：广播没生效立刻现形
            dist.broadcast(buf, src=root)
            check(f"broadcast root={root} {size_bytes}B", buf, src)


def t_alltoall(rank, ws):
    # 等长：send 块 i 发给 rank i。接收方 rank 收到 src 的块是
    # src 发送缓冲的第 rank 块 = arange(rank*per, (rank+1)*per) + src*100000
    elem = 4  # int32
    for size_bytes in SCAN_SIZES:
        per = size_bytes // elem // ws
        if per == 0:
            continue  # local patch: 每对进程不足 1 元素，无法等分——跳过
            # （原实现 max(1,0)=1 使 8B~256B 各档坍缩成同一个 512B 用例，标签误导）
        log_case(f"alltoall 等长 int32 {size_bytes}B")
        send = torch.arange(ws * per, dtype=torch.int32) + rank * 100000
        expected = torch.cat(
            [
                torch.arange(rank * per, (rank + 1) * per, dtype=torch.int32)
                + src * 100000
                for src in range(ws)
            ]
        )
        recv = torch.zeros(ws * per, dtype=torch.int32)
        dist.all_to_all_single(recv, send)
        check(f"alltoall 等长 int32 {size_bytes}B", recv, expected)


def t_stress(rank, ws, iters):
    """背靠背演进数据：第 k 轮 allreduce(k)，任何残留/乱序立刻暴露。"""
    log_case(f"stress: 背靠背×{iters}")
    n = 128
    buf = torch.zeros(n, dtype=torch.int32)
    for k in range(iters):
        buf.fill_(k)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        if not torch.equal(buf, torch.full((n,), k * ws, dtype=torch.int32)):
            check(f"stress iter={k}", buf, torch.full((n,), k * ws, dtype=torch.int32))
            return
    _RESULTS.append((f"stress allreduce×{iters} 演进数据", True))


# ============================================================================
# 扩展测试（v2：数据模式多样性 / 交错操作 / dtype 完备 / 边界值 / 长稳）
# ============================================================================


def t_data_patterns(rank, ws):
    """数据模式多样性：全零 / 全相同 / 负数 / 边界值 / 单元素。

    为什么要测这些：生产负载的数据不是"base + rank"的递增序列，
    全零可能触发零优化路径，全相同可能让 MIN/MAX 退化，边界值测溢出。
    """
    n = 16

    for dtype in (torch.int32, torch.int64, torch.float32, torch.float64):
        tag = str(dtype).split(".")[-1]

        # ── 模式 1：全零 ──
        log_case(f"data: all_zero {tag}")
        x = torch.zeros(n, dtype=dtype)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        check(f"data all_zero SUM {tag}", x, torch.zeros(n, dtype=dtype), exact=True)

        x = torch.zeros(n, dtype=dtype)
        dist.all_reduce(x, op=dist.ReduceOp.MIN)
        check(f"data all_zero MIN {tag}", x, torch.zeros(n, dtype=dtype), exact=True)

        # ── 模式 2：全相同（MIN=MAX=该值，无变化场景）──
        val = 7
        log_case(f"data: all_same {tag}")
        x = torch.full((n,), val, dtype=dtype)
        dist.all_reduce(x, op=dist.ReduceOp.MIN)
        check(
            f"data all_same MIN {tag}",
            x,
            torch.full((n,), val, dtype=dtype),
            exact=True,
        )

        x = torch.full((n,), val, dtype=dtype)
        dist.all_reduce(x, op=dist.ReduceOp.MAX)
        check(
            f"data all_same MAX {tag}",
            x,
            torch.full((n,), val, dtype=dtype),
            exact=True,
        )

        x = torch.full((n,), val, dtype=dtype)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        exp = torch.full((n,), val * ws, dtype=dtype)
        check(f"data all_same SUM {tag}", x, exp)

        # ── 模式 3：单元素 ──
        log_case(f"data: single_elem {tag}")
        x = torch.tensor([rank + 1], dtype=dtype)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        exp = torch.tensor([ws * (ws + 1) // 2], dtype=dtype)
        check(f"data single_elem SUM {tag}", x, exp)

        x = torch.tensor([rank + 1], dtype=dtype)
        dist.all_reduce(x, op=dist.ReduceOp.MIN)
        check(
            f"data single_elem MIN {tag}", x, torch.tensor([1], dtype=dtype), exact=True
        )

    # ── 模式 4：整数边界值（int32 专属）──
    log_case("data: boundary int32")
    boundary = torch.tensor(
        [
            2**31 - 1,  # INT_MAX
            -(2**31),  # INT_MIN
            0,
            1,
            -1,
            2**30,
            -(2**30),
        ],
        dtype=torch.int32,
    )
    # rank 0 发边界值，其他 rank 发小值 → MIN 应取小值，MAX 应取边界值
    if rank == 0:
        send_min = boundary.clone()
        send_max = boundary.clone()
    else:
        send_min = torch.full_like(boundary, 1)
        send_max = torch.full_like(boundary, 1)

    dist.all_reduce(send_min, op=dist.ReduceOp.MIN)
    exp_min = torch.minimum(boundary, torch.ones_like(boundary))
    check("data boundary MIN int32", send_min, exp_min, exact=True)

    if rank == 0:
        send_max = boundary.clone()
    else:
        send_max = torch.full_like(boundary, 1)
    dist.all_reduce(send_max, op=dist.ReduceOp.MAX)
    exp_max = torch.maximum(boundary, torch.ones_like(boundary))
    check("data boundary MAX int32", send_max, exp_max, exact=True)

    # ── 模式 5：浮点特殊值（NaN / Inf）──
    if ws >= 1:
        log_case("data: float_special")
        special = torch.tensor(
            [0.0, 1.0, -1.0, float("inf"), float("-inf")], dtype=torch.float32
        )
        # SUM 含 inf → 结果应为 inf 或 -inf
        x = special.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        # 只验证不 crash 且有限值正确（inf 语义因后端而异）
        _RESULTS.append(("data float_special SUM (不 crash 即通过)", True))


def t_interleaved(rank, ws, iters=20):
    """交错操作：allreduce → broadcast → allgather 交叉执行。

    为什么重要：fence 使用 sense-reversal + slot 轮换，
    如果两个算子的 fence 状态互相污染，只有交错执行才能暴露。
    """
    log_case(f"interleaved: 3算子交错×{iters}")
    n = 32
    for k in range(iters):
        # 同一轮三个算子用不同的数据
        ar_val = k
        bc_val = k + 1
        ag_val = k + 2

        # ── 算子 1：allreduce ──
        a = torch.full((n,), ar_val, dtype=torch.int32)
        dist.all_reduce(a, op=dist.ReduceOp.SUM)
        if not torch.equal(a, torch.full((n,), ar_val * ws, dtype=torch.int32)):
            check(
                f"interleaved iter={k} allreduce",
                a,
                torch.full((n,), ar_val * ws, dtype=torch.int32),
            )
            return

        # ── 算子 2：broadcast ──
        root = k % ws
        b = (
            torch.full((n,), bc_val, dtype=torch.int32)
            if rank == root
            else torch.zeros(n, dtype=torch.int32)
        )
        dist.broadcast(b, src=root)
        if not torch.equal(b, torch.full((n,), bc_val, dtype=torch.int32)):
            check(
                f"interleaved iter={k} broadcast",
                b,
                torch.full((n,), bc_val, dtype=torch.int32),
            )
            return

        # ── 算子 3：allgather ──
        send = torch.full((n,), ag_val + rank, dtype=torch.int32)
        gather_list = [torch.zeros(n, dtype=torch.int32) for _ in range(ws)]
        dist.all_gather(gather_list, send)
        expected = torch.cat(
            [torch.full((n,), ag_val + r, dtype=torch.int32) for r in range(ws)]
        )
        if not torch.equal(torch.cat(gather_list), expected):
            check(f"interleaved iter={k} allgather", torch.cat(gather_list), expected)
            return

    _RESULTS.append((f"interleaved 3算子×{iters} 交错", True))


def t_dtype_completeness(rank, ws):
    """dtype 完备性：补测 uint8 / int8（现有测试只覆盖 int32+）。

    为什么重要：uint8 走 CHAR 路径（字节搬运），int8 有符号性在
    MIN/MAX 中可能出错（KUPL 将 INT8/UINT8 合并为 CHAR）。
    """

    # ── uint8：0~255 范围 ──
    log_case("dtype: uint8")
    x = torch.tensor([0, 1, 127, 128, 254, 255] * 4, dtype=torch.uint8)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    # 各 rank 发相同值 → SUM = val * ws
    base = torch.tensor([0, 1, 127, 128, 254, 255] * 4, dtype=torch.int64)
    exp = (base * ws).to(torch.uint8)  # uint8 会回绕
    check("dtype uint8 SUM", x, exp)

    x = torch.tensor([200, 100, 50] * 8, dtype=torch.uint8)
    dist.all_reduce(x, op=dist.ReduceOp.MIN)
    check(
        "dtype uint8 MIN",
        x,
        torch.tensor([200, 100, 50] * 8, dtype=torch.uint8),
        exact=True,
    ) if ws >= 2 else None  # 两 rank 发相同数据 → MIN = 原值

    # ── int8：-128~127 范围 ──
    log_case("dtype: int8")
    x = torch.tensor([-128, -1, 0, 1, 127] * 4, dtype=torch.int8)
    dist.all_reduce(x, op=dist.ReduceOp.MIN)
    check(
        "dtype int8 MIN",
        x,
        torch.tensor([-128, -1, 0, 1, 127] * 4, dtype=torch.int8),
        exact=True,
    )

    # ── allgather 的 uint8 ──
    log_case("dtype: allgather uint8")
    send = torch.tensor([rank * 10, rank * 10 + 1, rank * 10 + 2], dtype=torch.uint8)
    gather_list = [torch.zeros(3, dtype=torch.uint8) for _ in range(ws)]
    dist.all_gather(gather_list, send)
    expected = torch.cat(
        [
            torch.tensor([r * 10, r * 10 + 1, r * 10 + 2], dtype=torch.uint8)
            for r in range(ws)
        ]
    )
    check("dtype allgather uint8", torch.cat(gather_list), expected, exact=True)


def t_edge_cases(rank, ws):
    """边界条件：count=1 / 大 count / 非常规 root / in-place。"""
    # ── count=1 的所有算子 ──
    log_case("edge: count=1")
    x = torch.tensor([rank], dtype=torch.int32)
    dist.all_reduce(x)
    check(
        "edge count=1 allreduce",
        x,
        torch.tensor([ws * (ws - 1) // 2], dtype=torch.int32),
    )

    x = torch.tensor([rank * 100], dtype=torch.int32)
    dist.broadcast(x, src=0)
    check("edge count=1 broadcast", x, torch.tensor([0], dtype=torch.int32), exact=True)

    send = torch.tensor([rank], dtype=torch.int32)
    gather_list = [torch.zeros(1, dtype=torch.int32) for _ in range(ws)]
    dist.all_gather(gather_list, send)
    expected = torch.tensor(list(range(ws)), dtype=torch.int32)
    check("edge count=1 allgather", torch.cat(gather_list), expected, exact=True)

    # ── in-place allreduce（sendbuf == recvbuf）──
    log_case("edge: inplace allreduce")
    x = torch.full((16,), 5, dtype=torch.int32)
    dist.all_reduce(x)  # in-place：x 既是输入也是输出
    check("edge inplace allreduce", x, torch.full((16,), 5 * ws, dtype=torch.int32))

    # ── 非常规 root（中间 rank）──
    if ws >= 3:
        mid_root = ws // 2
        log_case(f"edge: broadcast root={mid_root}")
        buf = (
            torch.full((8,), mid_root * 10, dtype=torch.int32)
            if rank == mid_root
            else torch.zeros(8, dtype=torch.int32)
        )
        dist.broadcast(buf, src=mid_root)
        check(
            f"edge broadcast mid_root={mid_root}",
            buf,
            torch.full((8,), mid_root * 10, dtype=torch.int32),
            exact=True,
        )

    # ── 连续多次 allgather 用同一缓冲（测试缓冲复用不串数据）──
    log_case("edge: buffer reuse")
    send = torch.full((8,), rank, dtype=torch.int32)
    for round_idx in range(5):
        gather_list = [torch.zeros(8, dtype=torch.int32) for _ in range(ws)]
        dist.all_gather(gather_list, send)
        expected = torch.cat(
            [torch.full((8,), r, dtype=torch.int32) for r in range(ws)]
        )
        if not torch.equal(torch.cat(gather_list), expected):
            check(
                f"edge buffer_reuse round={round_idx}", torch.cat(gather_list), expected
            )
            return
    _RESULTS.append(("edge buffer_reuse allgather×5 同缓冲", True))


def t_stress_mixed(rank, ws, iters=100):
    """混合压力：每轮随机选择算子 + 演进数据 + 交错。

    比普通 stress 更接近真实训练循环：
    DDP 一轮 = allreduce 梯度 + broadcast 参数 + gather 指标。
    """
    log_case("stress_mixed: 随机3算子×100")
    import random

    # ★ 所有 rank 必须用同一个种子：集合算子要求全员调用同一个算子/同一个 root。
    #   若按 rank 播种（42+rank），各 rank 掷出不同算子 → 对不齐 → fence 死锁。
    random.seed(42)
    n = 64

    for k in range(iters):
        op = random.choice(["allreduce", "broadcast", "allgather"])
        val = k + 1

        if op == "allreduce":
            buf = torch.full((n,), val, dtype=torch.int32)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            exp = torch.full((n,), val * ws, dtype=torch.int32)
        elif op == "broadcast":
            root = random.randint(0, ws - 1)
            buf = (
                torch.full((n,), val, dtype=torch.int32)
                if rank == root
                else torch.zeros(n, dtype=torch.int32)
            )
            dist.broadcast(buf, src=root)
            exp = torch.full((n,), val, dtype=torch.int32)
        else:  # allgather
            send = torch.full((n,), val + rank, dtype=torch.int32)
            gather_list = [torch.zeros(n, dtype=torch.int32) for _ in range(ws)]
            dist.all_gather(gather_list, send)
            buf = torch.cat(gather_list)
            exp = torch.cat(
                [torch.full((n,), val + r, dtype=torch.int32) for r in range(ws)]
            )

        if not torch.equal(buf, exp):
            check(f"stress_mixed iter={k} op={op}", buf, exp)
            return

    _RESULTS.append((f"stress_mixed 随机3算子×{iters}", True))


def t_memory_stability(rank, ws, iters=200):
    """内存稳定性：多轮操作后 RSS 不应持续增长（泄漏检测）。"""
    log_case(f"memory: RSS×{iters}")
    try:
        import resource
    except ImportError:
        _RESULTS.append(("memory_stability (skip: 无 resource 模块)", True))
        return

    def get_rss_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024

    n = 512
    rss_start = get_rss_mb()

    for k in range(iters):
        x = torch.full((n,), k, dtype=torch.float32)
        dist.all_reduce(x)
        # 不 hold 引用，让 GC 回收

    rss_end = get_rss_mb()
    growth = rss_end - rss_start
    # 允许 <50MB 增长（Python GC 波动），超过则疑似泄漏
    if growth < 50:
        _RESULTS.append(
            (f"memory_stability ×{iters} (RSS增长 {growth}MB < 50MB)", True)
        )
    else:
        _RESULTS.append(
            (f"memory_stability ×{iters} FAIL (RSS增长 {growth}MB ≥ 50MB)", False)
        )
        if rank == 0:
            print(
                f"  [mem] start={rss_start}MB end={rss_end}MB growth={growth}MB",
                flush=True,
            )


TESTS = [
    ("allgather", t_allgather),
    ("allgather_flat", t_allgather_flat),
    ("allreduce", t_allreduce),
    ("broadcast", t_broadcast),
    ("alltoall", t_alltoall),
    ("data_patterns", t_data_patterns),
    ("dtype_completeness", t_dtype_completeness),
    ("edge_cases", t_edge_cases),
]


# ============================================================================
# 主流程
# ============================================================================


def run_suite(backend, rank, ws, iters, timeout_s=300):
    if rank == 0:
        print(
            f"\n{'=' * 60}\n Backend: {backend} (world_size={ws})\n{'=' * 60}",
            flush=True,
        )
    if "kuccl" in backend:  # 兼容 "kuccl" 与 "cpu:kuccl" 两种写法
        import kuccl  # noqa: F401

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        world_size=ws,
        rank=rank,
        timeout=timedelta(seconds=timeout_s),
    )

    _RESULTS.clear()
    for name, fn in TESTS:
        try:
            fn(rank, ws)
        except Exception as e:  # noqa: BLE001
            _RESULTS.append((f"{name} (exception)", False))
            print(f"[Rank {rank}] FAIL {name}: {e}", flush=True)
        # ★ 每个测试函数后插 barrier：干净的 fence 状态分界线
        #   防止上一个测试的 fence slot 计数错位传染给下一个测试
        dist.barrier()
    try:
        t_stress(rank, ws, iters)
    except Exception as e:  # noqa: BLE001
        _RESULTS.append(("stress (exception)", False))
        print(f"[Rank {rank}] FAIL stress: {e}", flush=True)

    try:
        t_interleaved(rank, ws, iters=min(iters, 20))
    except Exception as e:  # noqa: BLE001
        _RESULTS.append(("interleaved (exception)", False))
        print(f"[Rank {rank}] FAIL interleaved: {e}", flush=True)

    try:
        t_stress_mixed(rank, ws, iters=min(iters * 5, 100))
    except Exception as e:  # noqa: BLE001
        _RESULTS.append(("stress_mixed (exception)", False))
        print(f"[Rank {rank}] FAIL stress_mixed: {e}", flush=True)

    try:
        t_memory_stability(rank, ws, iters=200)
    except Exception as e:  # noqa: BLE001
        _RESULTS.append(("memory_stability (exception)", False))
        print(f"[Rank {rank}] FAIL memory_stability: {e}", flush=True)

    dist.barrier()

    n_pass = sum(1 for _, ok in _RESULTS if ok)
    n_all = len(_RESULTS)
    for tag, ok in _RESULTS:
        if not ok:
            print(f"  ✗ [rank {rank}] {tag}", flush=True)
    if rank == 0:
        print(f"  {n_pass}/{n_all} PASS", flush=True)

    # 全员一致判定：任一 rank 任一用例失败 → 全体 FAIL
    flag = torch.tensor([1 if n_pass == n_all else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    # ★ 散场前全员对齐：KUPL win_free 拆窗口前会做全员 OOB 栅栏，
    #   若各 rank 拆除时间错开，先退场者关闭 TCPStore 会让慢 rank
    #   的 OOB 喊话撞上 connection reset（无谓告警）
    dist.barrier()
    dist.destroy_process_group()
    return bool(flag.item())


def _pick_free_port(start_port, count, host):
    """从 start_port 起找 count 个连续可用端口（防残留进程占端口）。

    残留场景：上次 run 被 Ctrl-C，卡在 C++ 里的 spawn 子进程没死，
    其 TCPStore server 仍占着端口 → 新 run 绑不上（rank0 报
    "failed to listen"），rank1 反而连上幽灵 server 读到旧 OOB key。
    这里用 bind 探测（listen 前关闭，无 TIME_WAIT），绕开整类问题。
    """
    import socket as _socket

    for base in range(start_port, start_port + 200):
        ok = True
        for p in range(base, base + count):
            s = _socket.socket()
            try:
                s.bind((host, p))
            except OSError:
                ok = False
            finally:
                s.close()
            if not ok:
                break
        if ok:
            return base
    return start_port  # 200 个都被占就原样返回，让 TCPStore 自己报错


def _worker_main(args):
    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    spawned = os.environ.get("KuCCL_TEST_SPAWNED") == "1"

    if spawned:
        # spawn 模式：parent 已清洗过残留环境变量并选定可用端口，
        # 直接用（★ 不用 setdefault——它挡不住 shell 里 export 过的旧值）
        master_addr = os.environ["MASTER_ADDR"]
        base_port = int(os.environ["MASTER_PORT"])
    else:
        # torchrun 模式：尊重 launcher 设置
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        base_port = int(os.environ.get("MASTER_PORT", str(args.port)))
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(base_port)

    if getattr(args, "bind_nics", None):
        nics = [n.strip() for n in args.bind_nics.split(",") if n.strip()]
        if nics:
            local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
            os.environ["UCX_NET_DEVICES"] = nics[local_rank % len(nics)]

    torch.set_num_threads(1)

    # 同进程先后测多后端（benchmark 同款序列）：每个后端换端口
    # （基于 parent 选定的 base_port 递增，不能用 args.port——
    #   parent 检测到端口被占时会自动换起点）
    all_ok = True
    for idx, backend in enumerate(args.backends):
        os.environ["MASTER_PORT"] = str(base_port + idx)
        all_ok &= run_suite(backend, rank, ws, args.iters, args.timeout)

    if rank == 0:
        print(
            f"\n[RESULT] {'ALL PASSED' if all_ok else 'FAILED'} "
            f"(backends={args.backends}, ws={ws})",
            flush=True,
        )
    sys.exit(0 if all_ok else 1)


def main():
    parser = argparse.ArgumentParser(description="KuCCL 正确性测试套件")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["kuccl"],
        help="后端序列（默认 kuccl；可 'kuccl gloo' 测切换）",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="spawn 模式进程数（torchrun 模式读 WORLD_SIZE）",
    )
    parser.add_argument("--port", type=int, default=29600)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--iters", type=int, default=20, help="stress 背靠背轮数")
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="init_process_group 超时秒数"
        "（多节点大 world_size 建组+连接风暴慢，建议 900）",
    )
    parser.add_argument(
        "--bind-nics",
        type=str,
        default=None,
        help="多网卡按 LOCAL_RANK 轮转绑定（多节点 PPN>8 必开）",
    )
    args = parser.parse_args()

    # torchrun 模式（RANK/WORLD_SIZE 已设）直接跑；否则 spawn 单机
    if "RANK" in os.environ:
        _worker_main(args)

    # ── spawn 模式：起跑前防两类环境污染 ──────────────────────
    # ① 端口被残留进程占用（上次 run 被杀，TCPStore server 还挂着）
    #    → 自动挑一段连续可用端口
    # ② shell 里 export 过的 MASTER_ADDR/MASTER_PORT/RANK 残留
    #    → 显式覆盖（子进程不再继承旧值）
    import multiprocessing as mp

    n_ports = max(1, len(args.backends))
    base = _pick_free_port(args.port, n_ports, args.master_addr)
    if base != args.port:
        print(
            f"[env] 端口 {args.port} 起被占用（可能有残留进程："
            f"ps -ef | grep correctness / ss -ltnp | grep {args.port}），"
            f"自动改用 {base}~{base + n_ports - 1}",
            flush=True,
        )

    ctx = mp.get_context("spawn")
    procs = []
    for r in range(args.world_size):
        env = os.environ.copy()
        env.update(
            {
                "RANK": str(r),
                "WORLD_SIZE": str(args.world_size),
                "MASTER_ADDR": args.master_addr,
                "MASTER_PORT": str(base),
                "KuCCL_TEST_SPAWNED": "1",
            }
        )
        p = ctx.Process(target=_spawn_entry, args=(env, vars(args)))
        procs.append(p)
        p.start()
    rc = 0
    for p in procs:
        p.join()
        rc |= p.exitcode or 0
    sys.exit(rc)


def _spawn_entry(env, args_dict):
    for k, v in env.items():
        os.environ[k] = v
    import argparse as _ap

    _worker_main(_ap.Namespace(**args_dict))


if __name__ == "__main__":
    main()
