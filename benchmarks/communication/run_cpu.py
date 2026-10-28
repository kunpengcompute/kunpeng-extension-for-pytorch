# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
Windows/CPU launcher for the communication benchmarks.

The benchmark scripts expect to be launched by ``deepspeed``/``mpirun``/``srun``
which set the distributed env variables.  On Windows CPU there is no such
launcher, so this script spawns ``--world-size`` child processes and hands each
one the correct ``RANK``/``WORLD_SIZE``/``LOCAL_RANK``/``MASTER_ADDR``/``MASTER_PORT``.

Usage::

    python run_cpu.py --world-size 2 -- all_reduce.py --dist torch --backend gloo --device cpu --scan --maxsize 16

Everything after ``--`` is passed through to the benchmark script.
"""

import argparse
import os
import subprocess
import sys


def _worker_env(i, world_size, master_addr, master_port):
    env = os.environ.copy()
    env["RANK"] = str(i)
    env["WORLD_SIZE"] = str(world_size)
    env["LOCAL_RANK"] = str(i)
    env["MASTER_ADDR"] = master_addr
    env["MASTER_PORT"] = master_port
    return env


def _parse_cpulist(s):
    """'0-3,8,10-11' → [0,1,2,3,8,10,11]"""
    cores = []
    for part in s.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            cores.extend(range(int(a), int(b) + 1))
        else:
            cores.append(int(part))
    return cores


def _cgroup_cpus():
    """cgroup cpuset 边界（sched_setaffinity 的真实约束）。
    注意它与 shell 亲和掩码是两回事——掩码可能只有几个散核，
    亲和性设置只受 cpuset 限制。读不到 → None（不约束）。"""
    for path in (
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpuset/cpuset.cpus",
    ):
        try:
            with open(path) as f:
                s = f.read().strip()
            if s:
                return set(_parse_cpulist(s))
        except OSError:
            continue
    return None


def _allowed_cpus():
    """绑核可用的核集 = cgroup cpuset（读不到时退回 shell 掩码，保守但不完美）。"""
    cg = _cgroup_cpus()
    return cg if cg is not None else set(os.sched_getaffinity(0))


def _node_cores(node):
    """NUMA 节点 node 的可绑核 = sysfs 物理拓扑 ∩ cgroup cpuset。
    交集对象必须是 cpuset 而非 sched_getaffinity(0)：后者可能只有
    个位数散核，用它求交会把节点砍剩 1 核，多 rank 挤单核 → 毫秒级栅栏。"""
    path = f"/sys/devices/system/node/node{node}/cpulist"
    with open(path) as f:
        cores = _parse_cpulist(f.read())
    allowed = _allowed_cpus()
    return [c for c in cores if c in allowed]


def _socket_of(cpu):
    """CPU 所属的 physical package（socket）id"""
    path = f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id"
    with open(path) as f:
        return int(f.read().strip())


def _bind_plan_sockets(world_size, sockets):
    """rank i → socket sockets[i % len(sockets)] 内的下一个可用核。
    每人独占一核（1:1，无超订），socket 间均分 rank → hier_socket 恒可用；
    若分到的核恰好聚在同一 die（≥2 人）die 级分组也成立，auto 自行路由。
    适用场景：cgroup 给的核散布在多个 NUMA 节点、
    每节点仅 1 核（hier_die 无法生效），但核分属 2 个 package——
    KUCCL_BIND_SOCKETS=0,1 即可测 socket 粒度分层。"""
    allowed = sorted(_allowed_cpus())
    by_sock = {}
    for c in allowed:
        by_sock.setdefault(_socket_of(c), []).append(c)
    plan = []
    idx = {s: 0 for s in sockets}
    for i in range(world_size):
        s = sockets[i % len(sockets)]
        cores = by_sock.get(s)
        if not cores:
            sys.exit(f"ERROR: no allowed CPUs on socket {s} (allowed cores: {allowed})")
        if idx[s] >= len(cores):
            sys.exit(
                f"ERROR: socket {s} has only {len(cores)} allowed cores for "
                f"{world_size // len(sockets)} ranks — oversubscription"
            )
        plan.append((i, s, [cores[idx[s]]]))
        idx[s] += 1
    return plan


def _all_nodes():
    """有可用核的 NUMA 节点，升序"""
    nodes = []
    for entry in os.listdir("/sys/devices/system/node"):
        if entry.startswith("node") and entry[4:].isdigit():
            n = int(entry[4:])
            if _node_cores(n):
                nodes.append(n)
    return sorted(nodes)


def _bind_plan_layout(world_size, layout):
    """仿 mpirun 的拓扑映射（每 rank 独占 1 核，等价 --bind-to core）：
      die:N    ≈ --map-by ppr:N:numa — node 升序，每 node 前 N 个核各放 1 rank
               （die:16=全塞首 die；die:8=node0:8+node1:8）
      socket   ≈ --map-by socket — np 均分到各 socket（socket 轮转，socket 内
               node 升序、核升序连续取），np=16 两 socket 时 = socket0 的 node0
               取 8 + socket1 的 node8 取 8
    每 rank 仅 1 核：布局精度优先；kupl 执行器线程 ≥256K 才并行（与 MPI 基准同条件）。"""
    allowed = _allowed_cpus()
    if layout.startswith("die:"):
        try:
            per = int(layout.split(":", 1)[1])
        except ValueError:
            sys.exit(
                f"ERROR: KUCCL_BIND_LAYOUT die:<N> needs an integer, got {layout!r}"
            )
        if per < 1:
            sys.exit(f"ERROR: KUCCL_BIND_LAYOUT die:<N> needs N >= 1, got {per}")
        plan = []
        for node in _all_nodes():
            for c in sorted(_node_cores(node))[:per]:
                if len(plan) >= world_size:
                    return plan
                plan.append((len(plan), node, [c]))
        if len(plan) < world_size:
            sys.exit(
                f"ERROR: layout die:{per} provides only {len(plan)} slots "
                f"for {world_size} ranks (allowed cores: {len(allowed)})"
            )
        return plan
    if layout == "socket":
        # 对齐 mpirun --map-by socket 的块均分语义：前 base+1 个 socket 各拿
        # base+1 个 rank（有余数时），socket 内 node 序、核序连续取——
        # ws=16 两 socket = socket0 连续 8 rank（node0 core0-7）+ socket1
        # 连续 8 rank（node8 core304-311）。不能轮转交替（实测与 mpirun
        # MISMATCH），也不能"填满一个 socket 再下一个"（小 np 全落首 socket）。
        by_sock = {}
        for node in _all_nodes():
            cores = _node_cores(node)
            by_sock.setdefault(_socket_of(cores[0]), []).extend(cores)
        sockets = sorted(by_sock)
        base, extra = divmod(world_size, len(sockets))
        plan = []
        for si, s in enumerate(sockets):
            take = base + (1 if si < extra else 0)
            if take > len(by_sock[s]):
                sys.exit(
                    f"ERROR: socket {s} has only {len(by_sock[s])} allowed cores "
                    f"for {take} ranks (ws={world_size}, {len(sockets)} sockets)"
                )
            for c in by_sock[s][:take]:
                plan.append((len(plan), s, [c]))
        return plan
    sys.exit(f"ERROR: KUCCL_BIND_LAYOUT must be 'die:<N>' or 'socket', got {layout!r}")


def _bind_plan(world_size, nodes):
    """rank i → 节点 nodes[i % len(nodes)] 的全部核（round-robin 保证均分）。
    节点粒度而非单核：核内留给 KUPL 执行器线程；节点内调度器自然均衡。"""
    plan = []
    for i in range(world_size):
        node = nodes[i % len(nodes)]
        cores = _node_cores(node)
        if not cores:
            sys.exit(f"ERROR: NUMA node {node} has no CPUs")
        plan.append((i, node, cores))
    return plan


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--world-size", type=int, default=2, help="number of ranks to spawn"
    )
    ap.add_argument("--master-addr", type=str, default="127.0.0.1")
    ap.add_argument("--master-port", type=str, default="29500")
    args, rest = ap.parse_known_args()

    if rest and rest[0] == "--":
        rest = rest[1:]

    if args.world_size < 2:
        # a single process would measure zero/trivial comm cost
        print(
            f"ERROR: --world-size must be >= 2 to measure communication, got {args.world_size}"
        )
        sys.exit(1)

    if not rest:
        print(
            'ERROR: pass a benchmark script after "--", e.g. "-- all_reduce.py --scan --maxsize 16"'
        )
        sys.exit(1)

    script = rest[0]
    passthrough = rest[1:]
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), script))
    if not os.path.exists(script_path):
        print(f"ERROR: benchmark script not found: {script_path}")
        sys.exit(1)

    # rank→NUMA 绑定（可选）。不绑时调度器随机撒 rank（16 rank 实测撒
    # 13 节点）；配对交换支持不均衡分布（每组 ≥2 人即可分层），绑定
    # 的收益是控制本地性、让流量分布可控。
    #   KUCCL_BIND_LAYOUT=die:N  每 die 前 N 核各放 1 rank（仿 mpirun ppr:N:numa）
    #   KUCCL_BIND_LAYOUT=socket socket 升序连续填满（仿 mpirun --map-by socket）
    #   KUCCL_BIND_NODES=0,1     die 粒度 round-robin，每 rank 拿整节点全部核
    #   KUCCL_BIND_SOCKETS=0,1   每 rank 独占一核、socket 间轮转
    # 仅对 kuccl 后端生效：gloo 进程设亲和性会崩溃（退出码 250），
    # 基线保持裸跑，对比才公平。
    backend = "kuccl"
    for a, v in zip(passthrough, passthrough[1:]):
        if a == "--backend":
            backend = v
            break
    bind_layout = None
    if os.environ.get("KUCCL_BIND_LAYOUT") and backend == "kuccl":
        bind_layout = os.environ["KUCCL_BIND_LAYOUT"].strip()
        if not bind_layout:
            sys.exit("ERROR: KUCCL_BIND_LAYOUT empty")
    bind_sockets = []
    if os.environ.get("KUCCL_BIND_SOCKETS") and backend == "kuccl" and not bind_layout:
        raw = os.environ["KUCCL_BIND_SOCKETS"]
        try:
            bind_sockets = [int(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            sys.exit(
                f"ERROR: KUCCL_BIND_SOCKETS must be comma-separated socket ids, got {raw!r}"
            )
        if not bind_sockets:
            sys.exit("ERROR: KUCCL_BIND_SOCKETS empty")
    bind_nodes = []
    if (
        os.environ.get("KUCCL_BIND_NODES")
        and backend == "kuccl"
        and not bind_sockets
        and not bind_layout
    ):
        raw = os.environ["KUCCL_BIND_NODES"]
        try:
            bind_nodes = [int(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            sys.exit(
                f"ERROR: KUCCL_BIND_NODES must be comma-separated node ids, got {raw!r}"
            )
        if not bind_nodes:
            sys.exit("ERROR: KUCCL_BIND_NODES empty")
    if bind_layout:
        plan = _bind_plan_layout(args.world_size, bind_layout)
        print(
            f"  [bind] layout={bind_layout}: "
            + ", ".join(
                f"rank{i}->{'node' if bind_layout.startswith('die') else 'sock'}{n}"
                for i, n, _ in plan[:8]
            )
            + (f" ... ({len(plan)} ranks)" if len(plan) > 8 else "")
        )
    elif bind_sockets:
        plan = _bind_plan_sockets(args.world_size, bind_sockets)
    else:
        plan = _bind_plan(args.world_size, bind_nodes) if bind_nodes else None
    if plan:
        for i, node, cores in plan:
            print(
                f"  [bind] rank {i} -> node {node} "
                f"(usable cores {len(cores)}, e.g. {min(cores)})"
            )
        if bind_nodes:
            ranks_per_node = args.world_size / len(bind_nodes)
            min_usable = min(len(cores) for _, _, cores in plan)
            if min_usable < ranks_per_node:
                print(
                    f"  [bind] WARNING: 每节点只有 {min_usable} 个可用核（cgroup cpuset 限制），"
                    f"{ranks_per_node} ranks/节点将共享单核——超订 + 自旋 = 毫秒级栅栏，"
                    f"拓扑基准无意义。请取消 KUCCL_BIND_NODES 或更换配额"
                    f"（同一 NUMA 节点连续多核）。"
                )
    elif backend != "kuccl" and (
        bind_layout
        or bind_sockets
        or bind_nodes
        or os.environ.get("KUCCL_BIND_LAYOUT")
        or os.environ.get("KUCCL_BIND_SOCKETS")
        or os.environ.get("KUCCL_BIND_NODES")
    ):
        print(f"  [bind] skipped (backend={backend}: binding is kuccl-only)")

    procs = []
    for i in range(args.world_size):
        cmd = [sys.executable, script_path, "--local_rank", str(i)] + passthrough
        env = _worker_env(i, args.world_size, args.master_addr, args.master_port)
        kwargs = {}
        if plan:
            cores = plan[i][2]
            # preexec_fn 在子进程 fork 后、exec 前执行：亲和性随 exec 保留
            kwargs["preexec_fn"] = lambda cs=set(cores): os.sched_setaffinity(0, cs)
        procs.append(subprocess.Popen(cmd, env=env, **kwargs))

    ret = 0
    for p in procs:
        rc = p.wait()
        if rc != 0:
            ret = rc
    sys.exit(ret)


if __name__ == "__main__":
    main()
