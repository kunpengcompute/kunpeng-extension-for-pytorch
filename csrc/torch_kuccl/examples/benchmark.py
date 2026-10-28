#!/usr/bin/env python3
"""
benchmark.py —— KUCCL vs Gloo 全算子性能对比（kuccl 真实场景口径）

性能对比：
    920F 单节点 PPN=N（推荐 8），8B~1M 包范围，kuccl vs gloo，目标提升 30%

测量方法（DeepSpeed 基准口径）：
    1. warmup 后 barrier 对齐起跑线（不计入）
    2. N 次背靠背算子调用（稳态流水，贴近 DDP 真实通信模式）
    3. 计时窗口只包含算子循环——尾部同步用 all_reduce(MAX) 完成，
       同时取"最慢 rank"的耗时作为该次集合通信的真实延迟
    4. barrier 自身成本不计入：两个后端 barrier 实现不同
       （gloo 原生网络 barrier vs kuccl TCPStore barrier），
       混入计时窗会系统性偏袒 gloo

用法：
    # 单机 2 进程（快速冒烟）
    python examples/benchmark.py

    # 单节点 PPN=8
    python examples/benchmark.py --world-size 8

    # 双节点 RoCE（真实跨机场景，两台机器都要执行）
    #   节点A: torchrun --nnodes=2 --nproc_per_node=8 --node-rank=0 \
    #            --master-addr=<节点A IP> --master-port=29500 examples/benchmark.py
    #   节点B: 同上，--node-rank=1
    #   （torchrun 会设置 RANK/WORLD_SIZE/MASTER_ADDR，脚本自动识别、不再 spawn）

    # 指定算子 / 规模 / 导出 CSV
    python examples/benchmark.py --ops allreduce --sizes 8 4096 262144 --csv r.csv

注意：
    - UCX 传输由环境变量控制（UCX_TLS / UCX_NET_DEVICES），开跑前打印实际值；
      单节点建议 UCX_TLS=cma,self（本 hucx 无 sm 组件；曾误写 sm），跨节点 RoCE 建议 UCX_NET_DEVICES=roh0（限单卡防连接爆炸）
    - kuccl 的 send/recv 是 alltoall 模拟（全员参与、无 tag），
      与 gloo 真 P2P 不同源，对比数据仅供参考
"""

import argparse
import os
import socket
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist


# ============================================================================
# 配置
# ============================================================================

# 8B ~ 1M，2 的幂递增（覆盖每个 octave）
DEFAULT_SIZES = [2**e for e in range(3, 21)]  # 8, 16, 32, ..., 1048576
DEFAULT_OPS = ["allgather", "allreduce", "broadcast", "alltoall"]
WARMUP_ITERS = 10
BENCH_ITERS = 50


def make_tensor(size_bytes, dtype=torch.float32):
    """按字节数创建 tensor（取最接近的元素数）"""
    elem_size = torch.tensor([], dtype=dtype).element_size()
    numel = max(1, size_bytes // elem_size)
    return torch.arange(numel, dtype=dtype)


# ============================================================================
# 计时核心：跨 rank 取最慢者，同步不进计时窗
# ============================================================================


def timed_collective(fn, iters):
    """跑 iters 次 fn，返回「最慢 rank」的平均单次耗时（ms）。

    尾部不进 barrier：用 all_reduce(MAX) 一并完成同步与最慢 rank 统计——
    集合通信的耗时由最慢参与者决定，单看 rank 0 本地时间会偏乐观。
    """
    dist.barrier()  # 起跑线对齐（不计入）
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()

    t = torch.tensor([end - start], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)  # 同步 + 取最慢 rank
    return t.item() / iters * 1000


# ============================================================================
# 各算子的 benchmark
# ============================================================================


def bench_allgather(rank, world_size, tensor, iters):
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    for _ in range(WARMUP_ITERS):
        dist.all_gather(gather_list, tensor)
    return timed_collective(lambda: dist.all_gather(gather_list, tensor), iters)


def bench_allgather_into_tensor(rank, world_size, tensor, iters):
    """all_gather_into_tensor（连续缓冲版，kuccl 零中转路径）"""
    recv = torch.empty(world_size * tensor.numel(), dtype=tensor.dtype)
    for _ in range(WARMUP_ITERS):
        dist.all_gather_into_tensor(recv, tensor)
    return timed_collective(lambda: dist.all_gather_into_tensor(recv, tensor), iters)


def bench_allreduce(rank, world_size, tensor, iters):
    buf = tensor.clone()
    for _ in range(WARMUP_ITERS):
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    return timed_collective(lambda: dist.all_reduce(buf, op=dist.ReduceOp.SUM), iters)


def bench_broadcast(rank, world_size, tensor, iters):
    buf = tensor.clone()
    for _ in range(WARMUP_ITERS):
        dist.broadcast(buf, src=0)
    return timed_collective(lambda: dist.broadcast(buf, src=0), iters)


def bench_alltoall(rank, world_size, tensor, iters):
    """等长 all_to_all_single"""
    if world_size < 2:
        return None
    per_peer = max(1, tensor.numel() // world_size)
    send = torch.arange(world_size * per_peer, dtype=tensor.dtype)
    recv = torch.empty(world_size * per_peer, dtype=tensor.dtype)
    for _ in range(WARMUP_ITERS):
        dist.all_to_all_single(recv, send)
    return timed_collective(lambda: dist.all_to_all_single(recv, send), iters)


def bench_send_recv(rank, world_size, tensor, iters):
    """点对点 ping-pong（kuccl 为 alltoall 模拟，gloo 真 P2P，仅供方向参考）"""
    if world_size < 2:
        return None
    peer = (rank + 1) % world_size
    send_buf = tensor.clone()
    recv_buf = torch.zeros_like(tensor)

    def _pingpong():
        if rank % 2 == 0:
            dist.send(send_buf, peer)
            dist.recv(recv_buf, peer)
        else:
            dist.recv(recv_buf, peer)
            dist.send(send_buf, peer)

    for _ in range(WARMUP_ITERS):
        _pingpong()
    return timed_collective(_pingpong, iters)


# ============================================================================
# 端到端（E2E）测试：用户视角——从初始化到完成的整个生命周期
#
# 现有算子级 benchmark 刻意排除建组/同步（测稳态算子性能）；E2E 刻意
# 全部包含：init_process_group（建组）→ 混合算子模拟训练负载 → 销毁。
# 用户感知的是"整个程序跑多久"，不是单个算子多快。
#
# 负载设计（模拟 DDP 一个训练 step 的通信形态）：
#   broadcast(参数) + allreduce(梯度)×3 + allgather + barrier
# 每轮一个 step，端到端计时从 init 前开始、destroy 后结束。
# 多轮 E2E 取平均，量纲 = "一次完整使用的总耗时"。
# ============================================================================


def run_e2e_once(backend, rank, world_size, steps, elem_counts):
    """一次完整生命周期：建组 → steps 轮混合负载 → 销毁。返回总耗时(ms)。"""
    t_start = time.perf_counter()

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=600),
    )
    t_init = time.perf_counter()

    # 模拟训练负载：每 step 的通信序列（DDP 形态）
    for _ in range(steps):
        # broadcast：参数同步（root=0）
        buf = torch.empty(elem_counts["bcast"], dtype=torch.float32)
        dist.broadcast(buf, src=0)
        # allreduce：梯度桶 ×3（不同大小，模拟分层梯度）
        for n in (
            elem_counts["grad_small"],
            elem_counts["grad_mid"],
            elem_counts["grad_large"],
        ):
            g = torch.ones(n, dtype=torch.float32)
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
        # allgather：状态收集
        a = torch.ones(elem_counts["gather"], dtype=torch.float32)
        out = [torch.empty_like(a) for _ in range(world_size)]
        dist.all_gather(out, a)
        # step 边界（同步点，真实训练里 optimizer.step 前的全局等待）
        dist.barrier()

    t_work = time.perf_counter()
    dist.destroy_process_group()
    t_end = time.perf_counter()

    return {
        "total_ms": (t_end - t_start) * 1000,  # 用户感知的全过程
        "init_ms": (t_init - t_start) * 1000,  # 建组（含连接风暴）
        "work_ms": (t_work - t_init) * 1000,  # 混合负载
        "teardown_ms": (t_end - t_work) * 1000,  # 销毁
    }


def bench_e2e(backend, rank, world_size, args):
    """多轮 E2E 取平均。返回分解后的平均耗时。

    ★ 计时口径与算子级 benchmark 相反：init/teardown 全部计入——
    用户视角没有"排除建组"这回事。init 占比本身就是重要指标
    （通信库的建组成本直接决定短任务的用户体验）。
    """
    elem_counts = {
        "bcast": 256 * 1024 // 4,  # 256KB 参数
        "grad_small": 64 * 1024 // 4,  # 64KB 梯度桶
        "grad_mid": 512 * 1024 // 4,  # 512KB
        "grad_large": 4 * 1024 * 1024 // 4,  # 4MB
        "gather": 128 * 1024 // 4,  # 128KB
    }
    rounds = max(1, args.e2e_rounds)
    steps = max(1, args.e2e_steps)

    # 首轮含进程冷启动（import 缓存/页表），discard 掉再计时
    run_e2e_once(backend, rank, world_size, steps=1, elem_counts=elem_counts)

    stats = {k: [] for k in ("total_ms", "init_ms", "work_ms", "teardown_ms")}
    for _ in range(rounds):
        r = run_e2e_once(backend, rank, world_size, steps, elem_counts)
        for k in stats:
            stats[k].append(r[k])

    # 跨 rank 取最慢者由调用方在独立进程组内 all_reduce(MAX) 聚合
    # （bench_e2e 自己的生命周期里进程组已销毁，无法内联聚合）
    out = {
        k: torch.tensor([sum(v) / len(v)], dtype=torch.float64)
        for k, v in stats.items()
    }
    return out


BENCH_FUNCS = {
    "allgather": bench_allgather,
    "allgather_flat": bench_allgather_into_tensor,
    "allreduce": bench_allreduce,
    "broadcast": bench_broadcast,
    "alltoall": bench_alltoall,
    "send_recv": bench_send_recv,
}


# ============================================================================
# 主流程
# ============================================================================


def run_benchmark(backend, ops, sizes, world_size, rank, csv_file=None):
    """跑一组 benchmark，返回结果列表"""
    results = []

    for op_name in ops:
        if op_name == "e2e":
            continue  # E2E 不是算子——主流程有独立分支处理（init+负载+销毁全计时）
        bench_fn = BENCH_FUNCS.get(op_name)
        if bench_fn is None:
            if rank == 0:
                print(f"[warn] 未知算子: {op_name}", flush=True)
            continue

        for size_bytes in sizes:
            tensor = make_tensor(size_bytes)

            try:
                latency_ms = bench_fn(rank, world_size, tensor, BENCH_ITERS)
            except Exception as e:
                if rank == 0:
                    print(f"[warn] {op_name} @ {size_bytes}B 失败: {e}", flush=True)
                continue

            if latency_ms is None:
                continue

            # 计算带宽
            actual_bytes = tensor.numel() * tensor.element_size()
            if op_name == "allgather":
                total_bytes = actual_bytes * world_size  # 每 rank 收全量
            elif op_name == "allreduce":
                total_bytes = actual_bytes * 2  # 发 + 收（近似）
            elif op_name == "broadcast":
                total_bytes = actual_bytes
            elif op_name == "alltoall":
                total_bytes = actual_bytes  # 近似
            elif op_name == "send_recv":
                total_bytes = actual_bytes * 2  # ping-pong
            else:
                total_bytes = actual_bytes

            bandwidth = total_bytes / (latency_ms / 1000.0) / (1024 * 1024)  # MB/s

            record = {
                "op": op_name,
                "backend": backend,
                "size_bytes": actual_bytes,
                "iters": BENCH_ITERS,
                "latency_ms": latency_ms,
                "bandwidth_MBps": bandwidth,
                "world_size": world_size,
            }
            results.append(record)

            if rank == 0:
                print(
                    f"  {op_name:16s} {actual_bytes:>8d}B  "
                    f"{latency_ms:10.4f} ms  {bandwidth:10.1f} MB/s",
                    flush=True,
                )

    return results


def main():
    global BENCH_ITERS, WARMUP_ITERS

    parser = argparse.ArgumentParser(description="KUCCL vs Gloo 全算子性能对比")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["kuccl", "gloo"],
        help="要测的后端（默认 kuccl gloo）",
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=DEFAULT_OPS,
        choices=list(BENCH_FUNCS.keys()) + ["e2e"],
        help="要测的算子",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=DEFAULT_SIZES, help="数据规模（字节数）"
    )
    parser.add_argument("--csv", type=str, default=None, help="输出 CSV 文件路径")
    parser.add_argument("--iters", type=int, default=BENCH_ITERS)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument(
        "--e2e-rounds",
        type=int,
        default=3,
        help="E2E 重复轮数（取平均；首轮冷启动已丢弃）",
    )
    parser.add_argument(
        "--e2e-steps", type=int, default=10, help="E2E 每轮模拟训练 step 数"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="进程数（spawn 模式用；torchrun 模式读 WORLD_SIZE）",
    )
    parser.add_argument(
        "--port", type=int, default=29500, help="master port（spawn 模式用）"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="init_process_group 超时秒数"
        "（多节点大 world_size 的建组+连接风暴较慢，建议 600+）",
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default="127.0.0.1",
        help="master 地址（spawn 模式用；跨节点请用 torchrun）",
    )
    parser.add_argument(
        "--bind-nics",
        type=str,
        default=None,
        help="多网卡按 rank 轮转绑定，如 'roh0,roh1,...,roh7'："
        "每个 rank 设 UCX_NET_DEVICES=nics[LOCAL_RANK%%n]，"
        "防止 PPN=16 时单卡 QP 爆炸（多节点必开）",
    )
    args = parser.parse_args()

    BENCH_ITERS = args.iters
    WARMUP_ITERS = args.warmup

    # 如果 RANK 环境变量不存在，说明是父进程 → spawn 子进程（单节点模式）
    if "RANK" not in os.environ:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        procs = []
        for r in range(args.world_size):
            env = os.environ.copy()
            env["RANK"] = str(r)
            env["WORLD_SIZE"] = str(args.world_size)
            env["MASTER_ADDR"] = args.master_addr
            env["MASTER_PORT"] = str(args.port)
            env["KUCCL_BENCH_SPAWNED"] = "1"
            p = ctx.Process(target=_spawn_worker, args=(env, vars(args)))
            procs.append(p)
            p.start()
        rc = 0
        for p in procs:
            p.join()
            rc |= p.exitcode or 0
        sys.exit(rc)

    # 子进程 / torchrun 进程走这里（RANK 已设置）
    _run_benchmark_main(args)


def _spawn_worker(env, args_dict):
    """spawn 出的子进程入口：设置环境变量后执行真正的 benchmark"""
    for k, v in env.items():
        os.environ[k] = v
    import argparse as _ap

    ns = _ap.Namespace(**args_dict)
    _run_benchmark_main(ns)


def _run_benchmark_main(args):
    """真正的 benchmark 主逻辑（在有 RANK/WORLD_SIZE 的进程中执行）"""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    spawned = os.environ.get("KUCCL_BENCH_SPAWNED") == "1"

    if spawned:
        # spawn 模式：用命令行参数（跨节点请走 torchrun，不要 spawn）
        master_addr = args.master_addr
        master_port = str(args.port)
    else:
        # torchrun 模式：尊重 launcher 设置的地址（跨节点 RoCE 的真实场景）
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # 计时稳定性：单线程 intra-op，避免多 rank 线程互相干扰
    # （真实部署用 numactl 绑核，脚本内先消除线程噪声）
    torch.set_num_threads(1)

    # 多网卡按 rank 轮转绑定（8 节点 PPN=16 防 QP 爆炸）：
    # 必须在 kuccl 首次 init_process_group（内部 ucp_init 读环境变量）之前生效。
    # gloo 不读 UCX_*，不受影响。
    if getattr(args, "bind_nics", None):
        nics = [n.strip() for n in args.bind_nics.split(",") if n.strip()]
        if nics:
            local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
            os.environ["UCX_NET_DEVICES"] = nics[local_rank % len(nics)]

    if world_size < 2 and "send_recv" in args.ops:
        args.ops.remove("send_recv")
    # e2e 与算子测试同跑时无需特殊处理（主流程已单独处理 e2e 分支）

    all_results = []

    for backend_idx, backend in enumerate(args.backends):
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f" Backend: {backend}  (world_size={world_size})")
            print(f"{'=' * 60}", flush=True)

        if backend == "kuccl":
            import kuccl  # noqa: F401

        # ★ 每个 backend 用不同端口（同一端口重建 TCPStore 会 Broken pipe）
        backend_port = int(master_port) + backend_idx
        os.environ["MASTER_PORT"] = str(backend_port)

        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_addr}:{backend_port}",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=args.timeout),
        )

        results = run_benchmark(backend, args.ops, args.sizes, world_size, rank)
        all_results.extend(results)

        dist.destroy_process_group()

        # ── E2E：用户视角全生命周期（init+负载+teardown 全计时）──
        # 必须在算子测试的进程组销毁之后跑（E2E 自己建组/销毁，端口独立）
        if "e2e" in args.ops:
            backend_port = int(master_port) + backend_idx + 100  # 避开算子端口
            os.environ["MASTER_PORT"] = str(backend_port)
            raw = bench_e2e(backend, rank, world_size, args)
            # 跨 rank 取最慢：需要进程组——用 E2E 最后一次建组已销毁，
            # 所以在 bench_e2e 内部最后一轮 destroy 之前无法聚合。
            # 改为：每轮结果存本地，这里用临时进程组聚合一次。
            dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{master_addr}:{backend_port + 1}",
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=args.timeout),
            )
            agg = {}
            for k, t in raw.items():
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                agg[k] = t.item()
            dist.destroy_process_group()

            e2e_record = {
                "op": "e2e",
                "backend": backend,
                "size_bytes": 0,  # E2E 无单一规模（混合负载）
                "iters": args.e2e_rounds * args.e2e_steps,
                "latency_ms": agg["total_ms"],
                "bandwidth_MBps": 0.0,
                "world_size": world_size,
                # 分解（同 CSV 导出，avg per round）
                "e2e_init_ms": agg["init_ms"],
                "e2e_work_ms": agg["work_ms"],
                "e2e_teardown_ms": agg["teardown_ms"],
            }
            all_results.append(e2e_record)
            if rank == 0:
                total = agg["total_ms"]
                print(
                    f"\n  ── E2E 用户视角（{args.e2e_rounds} 轮 × "
                    f"{args.e2e_steps} steps，含建组/销毁全计时）──",
                    flush=True,
                )
                print(f"  e2e_total    {total:10.2f} ms", flush=True)
                print(
                    f"    init       {agg['init_ms']:10.2f} ms "
                    f"({agg['init_ms'] / total * 100:4.1f}%)  ← 建组（连接风暴）",
                    flush=True,
                )
                print(
                    f"    work       {agg['work_ms']:10.2f} ms "
                    f"({agg['work_ms'] / total * 100:4.1f}%)  ← "
                    f"{args.e2e_steps} steps 混合负载",
                    flush=True,
                )
                print(
                    f"    teardown   {agg['teardown_ms']:10.2f} ms "
                    f"({agg['teardown_ms'] / total * 100:4.1f}%)",
                    flush=True,
                )

    # 环境与口径信息（写在结果后，方便截图汇报）
    if rank == 0:
        print(
            f"\n[env] host={socket.gethostname()} "
            f"UCX_TLS={os.environ.get('UCX_TLS', '<unset>')} "
            f"UCX_NET_DEVICES={os.environ.get('UCX_NET_DEVICES', '<unset>')} "
            f"KUCCL_BACKEND_TYPE={os.environ.get('KUCCL_BACKEND_TYPE', '<unset>')}"
        )
        print(
            "[口径] 延迟=跨 rank 最慢者；barrier/同步不在计时窗内；"
            "背靠背稳态（贴近 DDP）"
        )

    # 汇总对比（rank 0 打印）
    if rank == 0 and len(args.backends) == 2:
        print(f"\n{'=' * 60}")
        print(" 对比总结 (提升 = (gloo - kuccl) / gloo * 100%，目标 +30%)")
        print(f"{'=' * 60}")
        gloo_map = {
            (r["op"], r["size_bytes"]): r["latency_ms"]
            for r in all_results
            if r["backend"] == "gloo"
        }
        wins, total, speedups = 0, 0, []
        for r in all_results:
            if r["backend"] != "kuccl":
                continue
            g = gloo_map.get((r["op"], r["size_bytes"]))
            if g is None or g <= 0:
                continue
            k = r["latency_ms"]
            speedup = (g - k) / g * 100
            wins += 1 if speedup > 0 else 0
            total += 1
            speedups.append(speedup)
            if speedup >= 30:
                marker = "★"
            elif speedup > 0:
                marker = "✓"
            else:
                marker = "✗"
            print(
                f"  {r['op']:16s} {r['size_bytes']:>8d}B  "
                f"kuccl={k:8.4f}ms  gloo={g:8.4f}ms  "
                f"提升={speedup:+7.1f}% {marker}",
                flush=True,
            )
        if total:
            avg = sum(speedups) / len(speedups)
            print(
                f"\n  总计: {wins}/{total} 项 kuccl 快于 gloo，"
                f"平均提升 {avg:+.1f}%（目标 +30%）",
                flush=True,
            )

    # CSV 导出（e2e 记录附 init/work/teardown 分解列）
    if rank == 0 and args.csv and all_results:
        with open(args.csv, "w") as f:
            f.write(
                "op,backend,size_bytes,iters,latency_ms,"
                "bandwidth_MBps,world_size,"
                "e2e_init_ms,e2e_work_ms,e2e_teardown_ms\n"
            )
            for r in all_results:
                f.write(
                    f"{r['op']},{r['backend']},{r['size_bytes']},"
                    f"{r['iters']},{r['latency_ms']:.6f},"
                    f"{r['bandwidth_MBps']:.2f},{r['world_size']},"
                    f"{r.get('e2e_init_ms', '')},"
                    f"{r.get('e2e_work_ms', '')},"
                    f"{r.get('e2e_teardown_ms', '')}\n"
                )
        print(f"\n[ok] CSV 已保存: {args.csv}", flush=True)

    if rank == 0:
        print("\n[done] benchmark 完成", flush=True)


if __name__ == "__main__":
    main()
