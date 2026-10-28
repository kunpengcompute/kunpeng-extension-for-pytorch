#!/usr/bin/env python3
"""compare_run.py —— DeepSpeed communication 套件的双后端对比编排器。

零侵入设计：不 import / 不修改 communication/ 套件任何文件
（保留 DeepSpeed 上游的权威性），纯外部驱动：
  1. 以相同参数分别跑 <script>.py --backend kuccl 和 --backend gloo
     （各自经 run_cpu.py 拉起，同 world-size / trials / 规模扫描）
  2. 抓取 --raw 的机器可读输出（Size/Description/Duration(us)/Tput/BusBW）
  3. 按 (op, size) 对齐两份结果，计算提升：
        latency 提升 = (gloo - kuccl) / gloo * 100%   （与 examples/benchmark.py 同式）
        busbw  提升 = (kuccl - gloo)  / gloo  * 100%  （带宽越高越好，反向）
  4. 输出对比表：逐项标记（>=30%）、总计与平均（目标 +30%）

用法（在 benchmarks/communication/ 下）：
  # 默认：all_reduce 扫描 8B~1M，双后端对比
  python compare_run.py

  # 指定算子（可多个；pt2pt 只支持 ws=2）
  python compare_run.py --ops all_reduce all_gather broadcast

  # 规模与轮数
  python compare_run.py --maxsize 19 --trials 50 --warmups 5

  # 进程数 / 端口 / 额外透传参数（-- 之后原样传给套件脚本）
  python compare_run.py --world-size 4 -- --async-op --dtype float64

退出码：0 = 全部算子平均 latency 提升 >= gate；1 = 低于 gate 或运行失败。
"""

import argparse
import math
import re
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# 套件脚本名 → 算子名（raw 表头的 comm_op）
OP_SCRIPTS = [
    ("all_reduce", "all_reduce"),
    ("all_gather", "all_gather"),
    ("all_to_all", "all_to_all"),
    ("broadcast", "broadcast"),
    ("pt2pt", "pt2pt"),
]

# raw 行：Size Description Duration(us) Tput BusBW
# raw 模式 Size 列为纯字节数（无单位，如 "16.0"）；Description 形如 "4x4"
ROW_RE = re.compile(r"^\s*([\d.]+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")

UNIT = {"B": 1.0, "KB": 1e3, "MB": 1e6}


def size_to_bytes(num, unit):
    return float(num) * UNIT.get(unit or "B", 1.0)


def parse_raw_output(text, op):
    """从 run_cpu.py 的合并 stdout 里抠出 raw 表行 → {size_bytes: latency_us}。

    套件表格结构：
      ---- Performance of <op> on N devices ----
      Size (Bytes)  Description  Duration (us)  Throughput  BusBW   ← 列头
      ----------...----------                                    ← 表头分隔线
      16.0          4x4          1707.000        0.000        0.000 ← 数据行
      ----------...----------                                    ← 表尾分隔线
    分隔线出现两次：表头后（忽略）与表尾（终止本表）。
    判定规则：只有「已经匹配过至少一行数据」后遇到的分隔线才算表尾。
    """
    results = {}
    in_table = False
    saw_data = False
    for line in text.splitlines():
        if "Performance of" in line:
            # 只跟踪目标 op 的表（run_cpu 合并 stdout 可能含多算子输出）
            in_table = f"Performance of {op} " in line
            saw_data = False
            continue
        if not in_table:
            continue
        if set(line.strip()) <= {"-", " "} and line.strip():
            if saw_data:  # 表尾
                in_table = False
            continue  # 表头下的分隔线：忽略
        m = ROW_RE.match(line)
        if m:
            saw_data = True
            size_b = float(m.group(1))
            results[int(round(size_b))] = float(m.group(3))
    return results


def run_one(backend, script, args, passthrough):
    """跑一个后端一次，返回 raw 解析结果（失败返回 None + 错误输出尾部）。"""
    cmd = [
        sys.executable,
        os.path.join(HERE, "run_cpu.py"),
        "--world-size",
        str(args.world_size),
        "--master-addr",
        args.master_addr,
        "--master-port",
        str(args.port_for(backend)),
        "--",
        f"{script}.py",
        "--dist",
        "torch",
        "--backend",
        backend,
        "--device",
        "cpu",
        "--scan",
        "--maxsize",
        str(args.maxsize),
        "--trials",
        str(args.trials),
        "--warmups",
        str(args.warmups),
        "--raw",
    ] + list(passthrough)
    # OMP 限 1 线程（对齐 torchrun 生产口径）：OMP 默认全核时，多进程下
    # 线程栅栏成本会污染通信延迟（256K allreduce 虚增 10 倍）。
    # 显式设置过则尊重用户值。
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    print(f"  [run] backend={backend}: {' '.join(cmd[1:])}", flush=True)
    # errors='replace'：UCX/SDMA 的错误输出直写 fd，可能混入非 UTF-8
    # 字节，严格解码会在这里炸掉整个解析表。
    proc = subprocess.run(
        cmd,
        cwd=HERE,
        capture_output=True,
        text=True,
        timeout=args.timeout,
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        # gloo 大 ws 下有已知的收尾崩溃（terminate called without an active
        # exception，exit 250）——数据行已打完，属进程退出竞态，不挡对比。
        parsed = parse_raw_output(proc.stdout, script)
        if parsed:
            print(
                f"  [warn] {backend} 退出码 {proc.returncode}（收尾异常），"
                f"已解析 {len(parsed)} 个规模点，按现有数据继续"
            )
            return parsed
        print(f"  [error] {backend} 退出码 {proc.returncode}，输出尾部：")
        print("\n".join((proc.stdout + proc.stderr).splitlines()[-15:]))
        return None
    parsed = parse_raw_output(proc.stdout, script)
    if not parsed:
        print(f"  [error] {backend} 未解析到任何数据行，输出尾部：")
        print("\n".join(proc.stdout.splitlines()[-15:]))
    return parsed


def main():
    ap = argparse.ArgumentParser(
        description="DeepSpeed communication 套件双后端对比（零侵入，不改套件文件）"
    )
    ap.add_argument(
        "--ops",
        nargs="+",
        default=["all_reduce"],
        choices=[s for s, _ in OP_SCRIPTS],
        help="要对比的算子（默认 all_reduce）",
    )
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument(
        "--maxsize", type=int, default=19, help="扫描上限 2^N 字节（默认 20=1MB）"
    )
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--warmups", type=int, default=5)
    ap.add_argument("--master-addr", default="127.0.0.1")
    ap.add_argument(
        "--base-port",
        type=int,
        default=29710,
        help="端口起点（每个后端/算子自动递增，防幽灵占用）",
    )
    ap.add_argument("--timeout", type=int, default=600, help="单次运行超时（秒）")
    ap.add_argument(
        "--gate",
        type=float,
        default=30.0,
        help="判定阈值：平均 latency 提升 %（默认 30）",
    )
    args, passthrough = ap.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    # compare_run.py 自管后端（kuccl+gloo 各跑一遍，run_one 在命令行中段插入
    # --backend）。用户随手加的 --backend 会落进 passthrough 尾部，而 worker 的
    # argparse 对重复参数取最后一个 → gloo 列被静默替换成 kuccl 的数据。
    # 必须剥离，而不是透传。
    cleaned = []
    dropped = []
    skip_next = False
    for a in passthrough:
        if skip_next:
            skip_next = False
            continue
        if a == "--backend":
            skip_next = True  # 剥掉形参及其值
            dropped.append(a)
            continue
        if a.startswith("--backend="):
            dropped.append(a)
            continue
        cleaned.append(a)
    if dropped:
        print(
            f"  [warn] --backend 由本工具自管（双后端各跑一遍），透传参数中的 "
            f"{dropped} 已剥离（否则 gloo 列会被覆盖成 kuccl 数据）"
        )
    passthrough = cleaned

    port_counter = [args.base_port]
    args.port_for = lambda backend: (
        port_counter.append(port_counter[-1] + 1),
        port_counter[-2],
    )[1]

    print("═" * 72)
    print(
        f" DeepSpeed 套件双后端对比  ops={args.ops}  ws={args.world_size}  "
        f"maxsize=2^{args.maxsize}"
    )
    print("═" * 72)

    all_speedups = []
    _pair_lat = []  # (T_kuccl, T_gloo) 对——几何平均延迟比用
    wins = total = 0
    for script, op in OP_SCRIPTS:
        if script not in args.ops:
            continue
        print(f"\n──── {op} ────")
        k = run_one("kuccl", script, args, passthrough)
        g = run_one("gloo", script, args, passthrough)
        if k is None or g is None:
            print(f"  [{op}] 运行失败，跳过对比")
            continue

        common = sorted(set(k) & set(g))
        if not common:
            print(f"  [{op}] 两后端无对齐的规模点，跳过")
            continue
        n_ws = args.world_size
        print(
            f"  {'size':>10s} {'kuccl(us)':>11s} {'gloo(us)':>11s} "
            f"{'延迟提升':>9s} │ {'kuccl BusBW':>12s} {'gloo BusBW':>12s}"
        )
        print("  " + "-" * 88)
        # busbw = size/duration × 算法系数 × 8（bit），与套件 get_bw 完全同式：
        #   all_reduce/all_to_all 系数 2(n-1)/n；all_gather size×n 后同式；
        #   broadcast/pt2pt 系数 1
        ALGO = {
            "all_reduce": 2.0 * (n_ws - 1) / n_ws,
            "all_to_all": 1.0 * (n_ws - 1) / n_ws,
            "all_gather": 1.0 * (n_ws - 1) / n_ws,  # size 已 ×n（含在 size 里）
            "broadcast": 1.0,
            "pt2pt": 1.0,
        }
        algo = ALGO.get(op, 1.0)
        # all_gather 的 size 口径：套件 get_bw 里 size *= n——这里 size 列
        # 已是 input 总字节，直接用同式
        for size in common:
            kk, gg = k[size], g[size]  # µs
            if gg <= 0 or kk <= 0:
                continue
            # BusBW (Gbit/s) = bytes / (us×1e-6) × 算法系数 × 8bit / 1e9
            kb = size / (kk * 1e-6) * algo * 8 / 1e9
            gb = size / (gg * 1e-6) * algo * 8 / 1e9
            lat_up = (gg - kk) / gg * 100
            all_speedups.append(lat_up)
            _pair_lat.append((kk, gg))
            total += 1
            wins += 1 if lat_up > 0 else 0
            if lat_up >= args.gate:
                mark = "  ★"
            elif lat_up > 0:
                mark = "  ✓"
            else:
                mark = "  ✗"
            print(
                f"  {size:>9d}B {kk:>11.3f} {gg:>11.3f} "
                f"{lat_up:>+8.1f}% │ {kb:>10.3f}G {gb:>10.3f}G{mark}"
            )

    if not total:
        sys.exit("没有可对比的数据（检查上方错误）")

    # ── 双口径聚合（判定）─────────────────────────────
    # 主口径：算术平均延迟提升（对应"平均通信性能提升 30%"的减法语义）
    avg = sum(all_speedups) / len(all_speedups)
    # 辅口径：几何平均加速比（乘性指标，抗单档极值扭曲；性能论文惯例）
    #   geo_ratio = (Π T_gloo/T_kuccl)^(1/N) —— 呈现为"平均快 X.XX 倍"
    #   判定换算：加速比 ≥ 1.30 等价于延迟降 ≥ 30%（1/(1+0.30) ≈ 0.769 的倒数）
    log_sum = sum(math.log(g / k) for k, g in _pair_lat) if _pair_lat else 0.0
    geo_ratio = math.exp(log_sum / len(_pair_lat)) if _pair_lat else 1.0
    # 加速比→等效延迟提升换算（统一到 30% 语义）：提升 = 1 − 1/geo_ratio
    geo_equiv = (1.0 - 1.0 / geo_ratio) * 100 if geo_ratio > 0 else 0.0

    print("\n" + "═" * 72)
    print(f" 判定（8B~1M 平均通信性能提升 ≥{args.gate:.0f}%，≥ 者标 ★）")
    print("─" * 72)
    print(f"  逐项胜出     : {wins}/{total} 项 kuccl 快于 gloo")
    print(
        f"  主口径       : 算术平均延迟提升 {avg:+.1f}%"
        f"  → {'★' if avg >= args.gate else '✗'}"
    )
    print(
        f"  辅口径       : 几何平均加速比 {geo_ratio:.2f}x"
        f"  (等效延迟提升 {geo_equiv:+.1f}%)"
        f"  → {'★' if geo_equiv >= args.gate else '✗'}"
    )
    print("─" * 72)
    verdict = (avg >= args.gate) and (geo_equiv >= args.gate)
    print(f" 双口径一致 → {'★' if verdict else '✗'}")
    print("═" * 72)
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
