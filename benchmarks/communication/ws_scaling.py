#!/usr/bin/env python3
"""ws_scaling.py —— kupl 后端跨 world_size 性能实验 + matplotlib 可视化

目的：量化"规模病"——kupl 延迟随 world_size 的增长曲线 vs gloo。
计时口径：barrier→barrier 墙钟（rank0），包含慢到达者（汇合成本计入）。

用法（服务器，先 source 环境并完成两包构建）：
  cd benchmarks/communication
  python ws_scaling.py                       # 默认 ws=2,4,8,16 双后端
  python ws_scaling.py --world-sizes 2 4 8   # 自定义
  python ws_scaling.py --skip-gloo           # 只跑 kupl
输出：
  ws_scaling.csv        原始数据
  ws_scaling.png        对比图（4 算子 × 3 消息大小，log 纵轴）
  控制台                 增长因子表（ws16/ws2 延迟比——"规模病"的直接量化）
依赖：pip install matplotlib（缺库时脚本会提示）
"""

import argparse
import csv
import os
import time
from datetime import datetime

SIZES_BYTES = [4, 8192, 262144]  # 消息大小（字节）：小/中/大
OPS = ["all_reduce", "all_gather", "broadcast", "all_to_all"]


def worker(rank, ws, master_port, backend, trials, warmups, result_pipe):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(ws)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    if backend == "kuccl":
        import kuccl  # noqa: F401  auto-register

    dist.init_process_group(
        backend,
        rank=rank,
        world_size=ws,
        timeout=__import__("datetime").timedelta(seconds=300),
    )

    results = {}  # op -> {size: wall_us}
    for op in OPS:
        for nbytes in SIZES_BYTES:
            n = max(1, nbytes // 4)  # int32 元素数
            send = torch.arange(n * ws, dtype=torch.int32) % 100
            x = send[:n].clone()
            out = torch.empty(n * ws, dtype=torch.int32)

            def call_op():
                if op == "all_reduce":
                    dist.all_reduce(x)
                elif op == "all_gather":
                    dist.all_gather_into_tensor(out, x)
                elif op == "broadcast":
                    dist.broadcast(x, src=0)
                else:  # all_to_all
                    dist.all_to_all_single(out, send)

            for _ in range(warmups):
                call_op()
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(trials):
                call_op()
            dist.barrier()
            wall_us = (time.perf_counter() - t0) / trials * 1e6
            if rank == 0:
                results.setdefault(op, {})[nbytes] = wall_us

    dist.barrier()
    if rank == 0:
        result_pipe.send(results)
    dist.destroy_process_group()


def run_one(ws, backend, args, base_port):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    recv_pipe, send_pipe = ctx.Pipe(duplex=False)
    procs = []
    for r in range(ws):
        p = ctx.Process(
            target=worker,
            args=(r, ws, base_port, backend, args.trials, args.warmups, send_pipe),
        )
        p.start()
        procs.append(p)
    send_pipe.close()
    results = recv_pipe.recv()
    for p in procs:
        p.join()
    return results


def main():
    ap = argparse.ArgumentParser(description="kupl 跨 world_size 性能实验")
    ap.add_argument("--world-sizes", nargs="+", type=int, default=[2, 4, 8, 16])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--base-port", type=int, default=29800)
    ap.add_argument("--skip-gloo", action="store_true")
    args = ap.parse_args()

    # ★ matplotlib 延迟到绘图时才导入：环境缺库/损坏时实验照跑，数据照出 CSV
    backends = ["kuccl"] + ([] if args.skip_gloo else ["gloo"])
    data = {}  # backend -> ws -> op -> size -> wall_us
    for ws in args.world_sizes:
        for i, backend in enumerate(backends):
            port = args.base_port + ws * 10 + i
            print(
                f"▶ ws={ws:<3} backend={backend:<6} 跑 {len(OPS)}算子×{len(SIZES_BYTES)}大小"
                f"×{args.trials}轮 ...",
                flush=True,
            )
            t0 = time.perf_counter()
            data.setdefault(backend, {})[ws] = run_one(ws, backend, args, port)
            print(f"  完成，耗时 {time.perf_counter() - t0:.1f}s", flush=True)

    # ── CSV ──
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"ws_scaling_{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "world_size", "op", "size_bytes", "wall_us"])
        for bk, per_ws in data.items():
            for ws, per_op in per_ws.items():
                for op, per_size in per_op.items():
                    for nbytes, us in per_size.items():
                        w.writerow([bk, ws, op, nbytes, f"{us:.1f}"])

    # ── 增长因子表（规模病量化：latency(ws_max) / latency(2)）──
    print(
        f"\n{'=' * 76}\n 规模病量化：latency(ws={args.world_sizes[-1]}) / latency(ws={args.world_sizes[0]})\n{'=' * 76}"
    )
    print(f"{'op':<12}{'size':>8} │ " + "".join(f"{bk:>12}" for bk in data))
    ws_lo, ws_hi = args.world_sizes[0], args.world_sizes[-1]
    for op in OPS:
        for nbytes in SIZES_BYTES:
            row = f"{op:<12}{nbytes:>8} │ "
            for bk in data:
                try:
                    ratio = data[bk][ws_hi][op][nbytes] / data[bk][ws_lo][op][nbytes]
                    row += f"{ratio:>11.1f}x"
                except (KeyError, ZeroDivisionError):
                    row += f"{'n/a':>12}"
            print(row)

    # ── 图：4 算子子图，log 纵轴，kupl 实线 / gloo 虚线，颜色区分消息大小 ──
    # （缺 matplotlib 时跳过绘图——数据已在 CSV，装好后重跑或本地重画）
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            f"\n[提示] 无 matplotlib，跳过绘图；数据已存 {csv_path}"
            f"（pip install matplotlib 后重跑即可出图）",
            flush=True,
        )
        return

    colors = {
        SIZES_BYTES[0]: "tab:blue",
        SIZES_BYTES[1]: "tab:orange",
        SIZES_BYTES[2]: "tab:green",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"kuccl 跨 world_size 延迟（kupl vs gloo, trials={args.trials}, "
        f"{datetime.now().strftime('%F %T')}）",
        fontsize=13,
    )
    for ax, op in zip(axes.flat, OPS):
        for bk, ls in (("kuccl", "-"), ("gloo", "--")):
            if bk not in data:
                continue
            for nbytes in SIZES_BYTES:
                ys = [
                    data[bk].get(ws, {}).get(op, {}).get(nbytes)
                    for ws in args.world_sizes
                ]
                if any(y is None for y in ys):
                    continue
                label = f"{bk} {nbytes}B"
                ax.plot(
                    args.world_sizes,
                    ys,
                    ls,
                    marker="o",
                    color=colors[nbytes],
                    label=label,
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(args.world_sizes)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(op, fontsize=11)
        ax.set_xlabel("world_size")
        ax.set_ylabel("平均延迟 (us, log)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png_path = f"ws_scaling_{stamp}.png"
    fig.savefig(png_path, dpi=150)
    print(f"\n[输出] {png_path}\n[输出] {csv_path}", flush=True)


if __name__ == "__main__":
    main()
