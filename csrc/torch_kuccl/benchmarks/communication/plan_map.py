#!/usr/bin/env python3
"""plan_map.py —— 从 dsbench 日志提取「消息大小 → 算法」映射表

前提：跑分时带 --ucg-log-level info（UCG 每次 plan 选择打一行
"select plan 'X' in 'domain'"）。选择行在该尺寸的数据行之前打印，
属于该尺寸的 warmups+trials。

用法：
    python benchmarks/communication/plan_map.py            # 取最新 dsbench_kuccl_*.log
    python benchmarks/communication/plan_map.py <日志文件>

说明：
  - domain 前缀 'planm ucx' = planm(hicoll) 模块的 plan（如 allreduce 的 NAP）
  - 非 allreduce 段落里出现的 allreduce-domain 选择行是 sync_all barrier
    （1 字节 allreduce）的决策，不计入该算子，仅以 [barrier] 计数
"""
import glob
import time
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# domain → 算子归属
DOMAIN_OP = {"allreduce": "all_reduce", "allgatherv": "all_gather",
             "bcast": "broadcast", "alltoallv": "all_to_all"}
HDR = re.compile(r"Performance of (\w+) on")
SEL = re.compile(r"select (?:fallback )?plan '(.+?)' in '(planm ucx|planc ucx) (\w+)'")
ROW = re.compile(r"^(\d+)\s+\d+x\d+\s+([\d.]+)")


def list_candidates():
    """按修改时间倒序列出候选日志（新→旧），标注各自包含的算子。"""
    logs = glob.glob(os.path.join(HERE, "logs", "dsbench_kuccl_*.log"))
    logs.sort(key=os.path.getmtime, reverse=True)
    for i, p in enumerate(logs[:15], 1):
        ops = sorted({m.group(1) for line in open(p, encoding="utf-8", errors="replace")
                      if (m := HDR.search(line))})
        mt = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
        print(f"  [{i}] {mt}  {os.path.basename(p)}  ops={ops or ['-']}")
    return logs


def main():
    args = sys.argv[1:]
    logs = sorted(glob.glob(os.path.join(HERE, "logs", "dsbench_kuccl_*.log")),
                  key=os.path.getmtime, reverse=True)
    if args:
        arg = args[0]
        if arg.isdigit() and 1 <= int(arg) <= len(logs):
            path = logs[int(arg) - 1]
        else:
            path = arg
    elif logs:
        path = logs[0]  # 默认取最新
    else:
        sys.exit("未找到日志（先跑 compare_nodes 带 --ucg-log-level info）")
    if not os.path.isfile(path):
        sys.exit(f"日志不存在: {path}")
    print(f"# 来源: {path}\n# op          size(B)      us   algorithm")

    cur_op, pending, size = None, [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = HDR.search(line)
        if m:
            cur_op, pending = m.group(1), []
            continue
        m = SEL.search(line)
        if m:
            algo, mod, dom = m.group(1), m.group(2), m.group(3)
            pending.append((algo, mod, dom))
            continue
        m = ROW.match(line.strip())
        if m and cur_op:
            size, us = int(m.group(1)), float(m.group(2))
            parts = []
            for algo, mod, dom in pending:
                op = DOMAIN_OP.get(dom, dom)
                tag = f"{algo}[{mod}]" + ("" if op == cur_op else "[barrier]")
                if not parts or parts[-1].split("x")[0] != tag:
                    n = sum(1 for a, _, d in pending
                            if a == algo and d == dom)
                    parts.append(f"{tag}x{n}")
            seen, uniq = set(), []
            for p in parts:
                if p not in seen:
                    seen.add(p)
                    uniq.append(p)
            print(f"{cur_op:<12} {size:>9} {us:>9.1f}   {' | '.join(uniq)}")
            pending = []

    # 尾部提示其他候选（内容/时间不同，需要对照时用序号切换）
    others = [(i, p) for i, p in enumerate(logs, 1)
              if os.path.abspath(p) != os.path.abspath(path)]
    if others:
        print("\n# 其他候选（plan_map.py <序号> 切换）:")
        for i, p in others[:10]:
            mt = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
            ops = sorted({m.group(1) for line in open(p, encoding="utf-8", errors="replace")
                          if (m := HDR.search(line))})
            print(f"#   [{i}] {mt}  {os.path.basename(p)}  ops={ops or ['-']}")


if __name__ == "__main__":
    main()
