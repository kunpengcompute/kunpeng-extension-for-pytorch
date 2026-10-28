#!/usr/bin/env python3
"""compare_nodes.py —— DeepSpeed communication 套件的多节点双后端对比。

与 compare_run.py（单机版）同一判定口径，执行层不同：
  单机版 run_one = 本机 spawn run_cpu.py
  本文件         = SSH 到各节点并发拉起 torchrun → 套件脚本 run_all.py
                   （benchmarks/communication/ 上游原版，零修改）

流程：
  --gate  只跑正确性检查（各节点 torchrun tests/correctness.py），跑完退出
          不加 --gate = 直接跑分
  2. 清场：杀各节点残留 worker/agent（torchrun 偶发超时后挂起占端口）
  3. kuccl 轮 / gloo 轮：各节点并发 torchrun run_all.py --scan --raw
     （node0 的 stdout 落 master 日志，含 --raw 机器可读表）
  4. master 日志喂 compare_run.parse_raw_output → 按 (op, size) 对齐
     → 30% 判定（算术平均 + 几何加速比双口径，与 compare_run 同式）

用法（master 节点上执行；config/<档名>.sh 提供节点清单）：
  python compare_nodes.py --nnodes 8 --ppn 16 \
      --ops all_reduce all_gather all_to_all broadcast

退出码：0 = 正确性与双口径判定均通过；1 = 未通过/失败。
"""

import argparse
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent  # benchmarks/communication
sys.path.insert(0, str(HERE))
from compare_run import OP_SCRIPTS, parse_raw_output  # noqa: E402

SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
MASTER = ""
RC = 0


def ssh_run(ip, cmd, timeout=600):
    """SSH 执行并返回 (rc, 合并输出)；master（本机）直接 bash 执行。"""
    if ip == MASTER:
        r = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    else:
        r = subprocess.run(
            ["ssh", *SSH_OPTS, f"root@{ip}", cmd],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    return r.returncode, r.stdout + r.stderr


def get_self_ip():
    """本机 IP（hostname -I 首个；失败返回空串）"""
    try:
        return subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=5
        ).stdout.split()[0]
    except Exception:
        return ""


def expand_nodes(spec):
    """节点清单展开：支持紧凑写法 '前缀|起-止,起-止,单值'（空格容忍），
    普通 IP 字符串原样保留。例如：
        "29.204.23.|50-57,60-67"  →  16 个 IP
        "29.204.23.|70"           →  1 个 IP
    """
    out = []
    for item in spec:
        if "|" not in item:
            out.append(item)
            continue
        prefix, ranges = item.split("|", 1)
        prefix = prefix.strip()
        for part in ranges.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                if "-" in part:
                    lo, hi = part.split("-", 1)
                    lo, hi = int(lo), int(hi)
                    if lo > hi:
                        sys.exit(f"✗ 节点区间倒序: {item!r} 中的 {part}")
                    out.extend(f"{prefix}{i}" for i in range(lo, hi + 1))
                else:
                    out.append(f"{prefix}{int(part)}")
            except ValueError:
                sys.exit(
                    f"✗ 节点写法无法解析: {item!r} 中的 {part!r}"
                    "（期望 '前缀|起-止,单值' 形式）"
                )
    return out


def load_nodes(conf_name, nnodes):
    """加载 config/<名>.py（Python 配置档）：NODES/REPO_DIR 等。"""
    import importlib.util

    conf = Path(conf_name)
    if not conf.is_file():
        conf = HERE / "config" / f"{conf_name}.py"
    if not conf.is_file():
        sys.exit(f"✗ 集群配置不存在: {conf}")
    spec = importlib.util.spec_from_file_location("kuccl_cluster_conf", conf)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nodes = expand_nodes(mod.NODES)[:nnodes]
    if len(nodes) < nnodes:
        sys.exit(f"✗ 配置档节点数不足: {len(nodes)} < {nnodes}")
    for n in nodes:
        if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", n):
            sys.exit(f"✗ 展开后不是合法 IPv4: {n!r}（检查 NODES 写法）")
    return nodes, mod


def node_cmd(
    ip, master, repo, rank, nnodes, ppn, port, script, script_args, env_exports
):
    """单节点命令：激活环境 → 注入库开关 → torchrun 拉起套件脚本。"""
    exports = "".join(f"export {k}={v};" for k, v in env_exports.items())
    entry = (
        "benchmarks/communication/dsbench_entry.py"
        if "run_all.py" in script
        else script
    )
    return (
        f"cd '{repo}/torch_kuccl' && "
        f"source ../kuccl/kuccl_env.sh >/dev/null 2>&1 && "
        f"source ../env.sh >/dev/null 2>&1 && {exports} "
        f"torchrun --nnodes={nnodes} --nproc_per_node={ppn} "
        f"--node_rank={rank} --master_addr={master} --master_port={port} "
        f"{entry} {script_args}"
    )


def _clean_banner(out):
    """过滤 ssh 登录横幅（"Authorized users only..."）与空行——横幅来自
    stderr 且拼接在 stdout 之后，会把 READY/BROKEN 之类的末行判定顶掉"""
    return "\n".join(
        ln
        for ln in out.splitlines()
        if ln.strip() and "Authorized users" not in ln and "activities may" not in ln
    )


def nic_retrans(ip, nics):
    """ethtool -S 重传计数快照（只列非零项；用于孤立尺寸尖峰的丢包定罪）"""
    nics = " ".join(nics)
    cmd = (
        f"for n in {nics}; do ethtool -S $n 2>/dev/null | "
        f"grep -iE 'retrans|retry' | grep -vE ':\\s*0+$' | sed \"s/^/[$n] /\"; "
        f"done; true"
    )
    rc, out = ssh_run(ip, cmd, timeout=60)
    return _clean_banner(out).strip()


def node_health(ip):
    """节点健康快照：loadavg 三列 + 已知残留进程数（[x] 写法免疫自匹配）
    + CPU 大户（load 异常时直接看到是谁在烧）"""
    # 注：pgrep -c 无匹配时打印 0 且退出码 1，不要再 || echo 0（会出双 0）
    cmd = (
        "cat /proc/loadavg; "
        "pgrep -fc '[d]sbench_entry|[t]orch.distributed.run|[s]pawn_main'; true; "
        "ps -eo pcpu,comm --sort=-pcpu --no-headers | "
        'awk \'$2!="ps"{print "top=" $1 "% " $2; exit}\''
    )
    rc, out = ssh_run(ip, cmd, timeout=15)
    lines = [ln.strip() for ln in _clean_banner(out).splitlines() if ln.strip()]
    load = lines[0] if lines else "?"
    leftover = lines[1] if len(lines) > 1 else "?"
    top = lines[2] if len(lines) > 2 else "?"
    return f"load={load} leftover={leftover} {top}"


def launch(backend, args, nodes, repo, conf, port, tls):
    """一轮套件跑分：各节点并发 torchrun；node0 stdout 落 master 日志。"""
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / f"dsbench_{backend}_tls-{tls}_{time.strftime('%m%d_%H%M%S')}.log"
    suite_args = (
        f"--dist torch --backend {backend} --scan "
        f"--maxsize {args.maxsize} --trials {args.trials} "
        f"--warmups {args.warmups} --raw " + " ".join(args.ops_as_flags)
    )
    exports = {
        "TORCH_KUCCL_FLUSH": "1" if args.flush else "",
        "KUCCL_PROFILE": "1" if args.profile else "",
        "KUCCL_PERSISTENT_REQ": "1" if args.persist else "",
        "KUCCL_TOPOLOGY": "0" if args.no_topology else "",
        "KUCCL_LOG_LEVEL": args.log_level,
        "KUCCL_LOG_RANKS": args.log_ranks,
        "UCX_TLS": tls,
    }
    if args.ucg_log_level:
        exports["UCG_LOG_LEVEL"] = args.ucg_log_level
    for kv in args.env:
        if "=" not in kv:
            sys.exit(f"✗ --env 期望 KEY=VALUE 形式: {kv!r}")
        k, v = kv.split("=", 1)
        exports[k.strip()] = v.strip()
    if args.proxy:
        exports["http_proxy"] = args.proxy
        exports["https_proxy"] = args.proxy
        exports["no_proxy"] = "127.0.0.1,localhost"
    print(
        f"── [dsbench] backend={backend} trials={args.trials} "
        f"maxsize=2^{args.maxsize} port={port} ──"
    )
    injected = " ".join(f"{k}={v}" for k, v in exports.items() if v)
    if injected:
        print(f"  [env] {injected}")
    print("── [nic-retrans before] ──")
    for ip in nodes:
        out = nic_retrans(ip, conf.RETRANS_NICS)
        if out:
            print(f"[{ip}] {out}")
    t_retrans = time.time()
    failures = []

    def ignite(ip, rank):
        cmd = node_cmd(
            ip,
            MASTER,
            repo,
            rank,
            args.nnodes,
            args.ppn,
            port,
            "run_all.py",
            suite_args,
            exports,
        )
        is_master = ip == MASTER
        print(f"  [ignite] {ip} (node_rank={rank}) 启动 ...", flush=True)
        try:
            # 全部节点输出统一落同一日志（此前非 master 节点是 DEVNULL 黑箱）
            with open(log_path, "a", encoding="utf-8", errors="replace") as f:
                proc = subprocess.run(
                    ["ssh", *SSH_OPTS, f"root@{ip}", cmd]
                    if not is_master
                    else ["bash", "-c", cmd],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=args.round_timeout,
                )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9  # 超时：已落日志的 raw 表仍可解析（tar 不丢数据）
        if rc != 0:
            failures.append(ip)
        print(f"  [{backend}] {ip} exit={rc}{' (TIMEOUT)' if rc == -9 else ''}")

    threads = []
    for rank, ip in enumerate(nodes):
        t = threading.Thread(target=ignite, args=(ip, rank))
        t.start()
        threads.append(t)
    t0 = time.time()
    hb_stop = threading.Event()

    def heartbeat():
        """rendezvous 看门狗：每 60s 报 master 端口状态 + agent 存活数，
        卡 5 分钟自动打印判读提示（此前只有"已等 Ns"，卡死时两眼一抹黑）"""
        beat = 0
        while not hb_stop.wait(60):
            if not any(t.is_alive() for t in threads):
                continue  # 轮次已收尾（线程退出在即），别再打误导性心跳
            beat += 1
            wait_s = int(time.time() - t0)
            try:
                _, out = ssh_run(
                    MASTER, f"ss -tln | grep -E ':{port}\\b' || true", timeout=10
                )
                listening = bool(_clean_banner(out).strip())
            except Exception:
                listening = None
            msg = (
                f"[wait] t+{wait_s}s master:{port} "
                f"{'LISTEN ✓' if listening else ('未监听 ⚠' if listening is False else '探测失败')}"
            )
            if beat % 2 == 0:  # 每 2 拍扫一遍 agent 存活（省 8 次 ssh/分钟）
                try:
                    with ThreadPoolExecutor(max_workers=len(nodes)) as ex:
                        res = list(
                            ex.map(
                                lambda ip: ssh_run(
                                    ip,
                                    "pgrep -c -f '[t]orch.distributed.run'; true",
                                    timeout=10,
                                ),
                                nodes,
                            )
                        )
                    dead = []
                    for ip, (_, out) in zip(nodes, res):
                        first = _clean_banner(out).strip().splitlines()
                        if not (first and first[0].isdigit() and int(first[0]) > 0):
                            dead.append(ip.split(".")[-1])
                    msg += f" | agent存活 {len(nodes) - len(dead)}/{len(nodes)}"
                    if dead:
                        msg += f"（缺席: .{','.join(dead)}）"
                except Exception:
                    pass
            if wait_s >= 300 and listening is False:
                msg += (
                    "\n         ⚠ 5 分钟 master 端口仍未监听——查 master 负载"
                    "(/proc/loadavg)、自机解析(getent hosts)、残留占用(ss -tlnp)"
                )
            elif wait_s >= 300 and listening:
                msg += (
                    "\n         ⚠ 端口已监听但未开跑——worker 连不上(测 "
                    "</dev/tcp/master/port)或上述缺席节点的日志"
                )
            print(msg, flush=True)

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    for t in threads:
        t.join()
    hb_stop.set()
    if failures:
        print(f"  ⚠ {backend} 轮失败节点: {failures}")
    print("── [nic-retrans after] ──")
    for ip in nodes:
        out = nic_retrans(ip, conf.RETRANS_NICS)
        if out:
            print(f"[{ip}] {out}  (t={time.time() - t_retrans:.0f}s since before)")
    rc, text = ssh_run(MASTER, f"cat '{log_path}'", timeout=60)
    return text


def correctness_gate(args, nodes, repo, conf, tls, gate_timeout):
    """正确性检查：node0 日志出现 FAIL 则不通过。"""
    print("── [gate] correctness（torchrun tests/correctness.py）──")
    logs = HERE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"gate_tls-{tls}_{time.strftime('%m%d_%H%M%S')}_{os.getpid()}.log"
    failures = []

    def run(ip, rank):
        gate_exports = {"UCX_TLS": tls}
        for kv in args.env:
            if "=" in kv:
                k, v = kv.split("=", 1)
                gate_exports[k.strip()] = v.strip()
        cmd = node_cmd(
            ip,
            MASTER,
            repo,
            rank,
            args.nnodes,
            args.ppn,
            args.port - 1,
            "tests/correctness.py",
            "--iters 20 --timeout 900",
            gate_exports,
        )
        is_master = ip == MASTER
        try:
            with open(log_path, "a", encoding="utf-8", errors="replace") as f:
                proc = subprocess.run(
                    ["ssh", *SSH_OPTS, f"root@{ip}", cmd]
                    if not is_master
                    else ["bash", "-c", cmd],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=gate_timeout,
                )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9  # 超时：ssh 已杀，远端残留由下轮清场兜底
        if rc != 0:
            failures.append(ip)
        print(f"  [gate] {ip} exit={rc} {'(TIMEOUT)' if rc == -9 else ''}")

    threads = []
    for rank, ip in enumerate(nodes):
        t = threading.Thread(target=run, args=(ip, rank))
        t.start()
        threads.append(t)
    t0 = time.time()
    hb_stop = threading.Event()

    def heartbeat():
        while not hb_stop.wait(60):
            print(f"[gate][wait] 已等 {int(time.time() - t0)}s ...", flush=True)

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    for t in threads:
        t.join()
    hb_stop.set()
    text = log_path.read_text(encoding="utf-8", errors="replace")
    n_fail = len(re.findall(r"\bFAIL\b", text))
    if failures or n_fail:
        print(f"  ✗ 未通过（FAIL 行 {n_fail}，失败节点 {failures}）")
        return False
    print("  ✓ 通过")
    return True


def main():
    ap = argparse.ArgumentParser(
        prog="compare_nodes.py",
        description="DeepSpeed 套件多节点双后端对比（复用 compare_run 判定口径）",
    )
    ap.add_argument("--nnodes", type=int, default=8)
    ap.add_argument("--ppn", type=int, default=16)
    ap.add_argument("--conf", default="default", help="集群配置档 config/<名>.sh")
    ap.add_argument(
        "--ops",
        nargs="+",
        default=[s for s, _ in OP_SCRIPTS if s != "pt2pt"],
        help="算子（默认四集合算子；pt2pt 不支持）",
    )
    ap.add_argument(
        "--maxsize",
        type=int,
        default=15,
        help="扫描上限 2^N 字节（默认 15=32MB/rank 输入；ws=128 时"
        " allgather 输出缓冲随 ws 放大，设太大必 OOM）",
    )
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--warmups", type=int, default=5)
    ap.add_argument(
        "--min-gain",
        type=float,
        default=30.0,
        help="判定阈值：平均延迟提升 %%（默认 30）",
    )
    ap.add_argument(
        "--backends",
        default="kuccl,gloo",
        help="参与跑分的后端（逗号分隔，默认 kuccl,gloo；算法横评用 kuccl）",
    )
    ap.add_argument(
        "--gate-timeout",
        type=int,
        default=1200,
        help="正确性检查单节点超时秒数（默认 1200，超时记失败并清场）",
    )
    ap.add_argument("--gate", action="store_true", help="只跑正确性检查（不跑分）")
    ap.add_argument("--port", type=int, default=29510, help="rendezvous 端口起点")
    ap.add_argument(
        "--no-clean",
        action="store_true",
        help="跳过清场 pkill（默认每轮前清理上轮残留）",
    )
    ap.add_argument("--flush", action="store_true", help="TORCH_KUCCL_FLUSH=1")
    ap.add_argument("--profile", action="store_true", help="KUCCL_PROFILE=1")
    ap.add_argument("--persist", action="store_true", help="KUCCL_PERSISTENT_REQ=1")
    ap.add_argument(
        "--no-topology",
        action="store_true",
        help="KUCCL_TOPOLOGY=0（拓扑喂入默认开；此旗标关掉做平铺 A/B）",
    )
    ap.add_argument(
        "--log-level",
        default="",
        help="KUCCL_LOG_LEVEL 透传（INFO/TRACE 等；空=默认 WARN）",
    )
    ap.add_argument(
        "--log-ranks",
        default="",
        help="KUCCL_LOG_RANKS 透传（如 0 或 0,3-5——大 world 排障限流；空=不过滤）",
    )
    ap.add_argument(
        "--ucg-log-level",
        default="",
        help="UCG_LOG_LEVEL 透传，仅全局 rank 0 生效（info 可看算法选择；debug 可看拓扑模型；空=默认 warn）",
    )
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="附加环境变量透传到 worker（可多次，如 --env UCX_RNDV_THRESH=2097152）",
    )
    ap.add_argument(
        "--round-timeout",
        type=int,
        default=3600,
        help="单后端轮超时秒数（默认 3600；超时时已写入日志的 raw 表仍可解析）",
    )
    ap.add_argument(
        "--tls",
        default="",
        help="UCX_TLS 覆盖（默认按节点数取 config 的 TLS_MULTI/TLS_SINGLE；"
        "显式指定时建议同时 KUCCL_TLS_USER=1 标记）",
    )
    ap.add_argument(
        "--sync",
        action="store_true",
        help="向各节点 rsync 分发代码（默认关：节点与开发机挂载同一"
        "共享存储，代码即写即见；仅节点本地盘部署时才需要）",
    )
    ap.add_argument(
        "--sync-mode",
        choices=["rsync", "git"],
        default="rsync",
        help="--sync 时的同步方式",
    )
    ap.add_argument(
        "--proxy",
        default="",
        help="节点临时代理（如 http://29.204.217.12:3128）——"
        "git/yum 可用；DNS 在代理端解析，本机无 DNS 也可",
    )
    ap.add_argument(
        "--local",
        action="store_true",
        help="本机执行模式。推荐始终加（共享挂载下代码即写即见，"
        "本机直控全部节点，零转发零分发）；不加则开发机自动转发 master",
    )
    args = ap.parse_args()
    args.ops_as_flags = ["--" + o.replace("_", "-") for o in args.ops]

    ws = args.nnodes * args.ppn
    nodes, conf = load_nodes(args.conf, args.nnodes)
    repo = conf.REPO_DIR
    global MASTER
    MASTER = nodes[0]

    # 开发机自动转发：执行机不在节点清单 → SSH 到 master（节点表第 1 台）分发
    # 代码并远程执行自身。同步 = rsync 增量（无 --delete，绝不删目标文件），
    # 远端 build.sh（kuccl 增量 / torch_kuccl 全量）。
    self_ip = get_self_ip()
    if self_ip not in nodes and not args.local:
        tk_dir = HERE.parents[1] / "csrc" / "torch_kuccl"  # 本仓库 csrc/torch_kuccl
        kuccl_lib = tk_dir.parent / "kuccl"  # <仓库根>/kuccl（库本体）
        if not (tk_dir / "kuccl" / "__init__.py").is_file() or not kuccl_lib.is_dir():
            sys.exit(f"✗ 目录推算异常: {tk_dir}（缺 kuccl/ 包或兄弟 kuccl/ 库）")
        # 单实例锁：并发实例的 rsync/跑分会互相干扰
        import fcntl

        _lock = open("/tmp/kuccl_compare_nodes.lock", "w")
        try:
            fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("✗ 已有 compare_nodes 实例在运行（/tmp/kuccl_compare_nodes.lock）")

        repo = conf.REPO_DIR
        # 目标根目录不存在时 rsync 直接失败（只建末级目录）——先 mkdir；
        # 根文件 env.sh 也随发（node_cmd 要 source ../env.sh，不在两个包目录里）
        root_env = tk_dir.parent / "env.sh"
        if not root_env.is_file():
            sys.exit(f"✗ 缺仓库根 env.sh: {root_env}")
        print(
            f"[dev] 执行机({self_ip})不在集群内——分发代码到 master({MASTER}) 并远程执行"
        )
        rc_mkdir, out_mkdir = ssh_run(MASTER, f"mkdir -p '{repo}'", timeout=15)
        if rc_mkdir != 0:
            sys.exit(f"✗ master 建目录失败: {out_mkdir[-400:]}")
        for src, dst in ((tk_dir, f"{repo}/torch_kuccl"), (kuccl_lib, f"{repo}/kuccl")):
            print(f"  [dev] rsync {os.path.basename(src)} → master ...", flush=True)
            r = subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--exclude=.git",
                    "--exclude=__pycache__",
                    "--exclude=logs",
                    f"{src}/",
                    f"root@{MASTER}:{dst}/",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if r.returncode != 0:
                sys.exit(f"✗ rsync {src} 失败: {r.stderr[-800:]}")
        r = subprocess.run(
            ["rsync", "-a", str(root_env), f"root@{MASTER}:{repo}/env.sh"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if r.returncode != 0:
            sys.exit(f"✗ rsync env.sh 失败: {r.stderr[-800:]}")
        print(
            "[dev] 代码+编译产物+根 env.sh 分发 ✓（build/ 随 rsync 带过去，免编译直接跑）"
        )
        remote_cmd = (
            f"cd '{repo}/torch_kuccl' && "
            f"python3 benchmarks/communication/compare_nodes.py --local "
            + " ".join(shlex.quote(a) for a in sys.argv[1:])
        )
        r = subprocess.run(["ssh", *SSH_OPTS, f"root@{MASTER}", remote_cmd])
        sys.exit(r.returncode)

    # TLS：默认按节点数取 config 的 TLS_MULTI / TLS_SINGLE；--tls 显式覆盖
    tls = args.tls or (conf.TLS_SINGLE if args.nnodes <= 1 else conf.TLS_MULTI)

    # 环境残留防线：执行机 shell 残留的 UCX_TLS 会经 ssh 继承到 worker——
    # 未带 KUCCL_TLS_USER 标记的残留值直接清掉（worker 由本编排显式注入）
    if os.environ.get("UCX_TLS") and not os.environ.get("KUCCL_TLS_USER"):
        print(
            f"[env] 忽略 shell 残留 UCX_TLS={os.environ['UCX_TLS']}"
            f"（未显式指定）→ 使用 {tls}"
        )
        del os.environ["UCX_TLS"]

    print("═" * 72)
    print(f" DeepSpeed 套件多节点对比  {args.nnodes} 节点 × PPN={args.ppn}  ws={ws}")
    print(
        f" ops={args.ops}  maxsize=2^{args.maxsize}  trials={args.trials}  "
        f"warmups={args.warmups}  UCX_TLS={tls}"
    )
    print("═" * 72)

    # 预检五段全并行（串行时 env-check 的 8 次 torch import 能拖 2 分钟）
    with ThreadPoolExecutor(max_workers=len(nodes)) as ex:
        pings = list(ex.map(lambda ip: (ip, *ssh_run(ip, "true", timeout=15)), nodes))
    for ip, rc, out in pings:
        if rc != 0:
            if "REMOTE HOST IDENTIFICATION HAS CHANGED" in out:
                sys.exit(
                    f"✗ {ip} 主机指纹变更（节点重装/换机？）——与管理员确认后"
                    f"执行 ssh-keygen -R {ip} 再重跑"
                )
            sys.exit(f"✗ {ip} 不可达（先配免密 SSH）")
    print(f"✓ {len(nodes)} 台连通")
    print("── [health] 节点负载/残留快照 ──")
    with ThreadPoolExecutor(max_workers=len(nodes)) as ex:
        snaps = list(ex.map(node_health, nodes))
    for ip, snap in zip(nodes, snaps):
        print(f"  [{ip}] {snap}")

    print("── [env-check] 逐节点环境就绪检查（挂载/conda/源/kuccl 库）──")

    def env_check_one(ip):
        # 逐项探测：不用 && 短路链——任何一项失败也把其余项查完，FAIL 项即问题点
        probe = (
            f"cd '{repo}/torch_kuccl' 2>/dev/null || true; "
            f"mount=$([ -f benchmarks/communication/compare_nodes.py ] "
            f"&& echo OK || echo FAIL); "
            f"source ../kuccl/kuccl_env.sh >/dev/null 2>&1; e1=$?; "
            f"source ../env.sh >/dev/null 2>&1; e2=$?; "
            f"command -v torchrun >/dev/null 2>&1; e3=$?; "
            f"python -c 'import torch' >/dev/null 2>&1; e4=$?; "
            f"python -c 'import kuccl' >/dev/null 2>&1; e5=$?; "
            f'ok() {{ [ "$1" -eq 0 ] && echo OK || echo FAIL; }}; '
            f'echo "items: mount=$mount kuccl_env=$(ok $e1) env_sh=$(ok $e2) '
            f'torchrun=$(ok $e3) torch=$(ok $e4) kuccl=$(ok $e5)"; '
            f'if [ "$mount" = OK ] && [ $e1 -eq 0 ] && [ $e2 -eq 0 ] && '
            f"[ $e3 -eq 0 ] && [ $e4 -eq 0 ] && [ $e5 -eq 0 ]; then echo READY; "
            f"else echo BROKEN; fi; true"
        )
        try:
            rc, out = ssh_run(ip, probe, timeout=90)
        except subprocess.TimeoutExpired:
            return ip, "STUCK(90s 无响应)", ""
        lines = [ln.strip() for ln in _clean_banner(out).splitlines() if ln.strip()]
        items = next((ln for ln in lines if ln.startswith("items:")), "")
        tag = (
            "READY"
            if "READY" in lines
            else (
                "BROKEN" if "BROKEN" in lines else (lines[-1] if lines else "(无输出)")
            )
        )
        return ip, tag, items

    with ThreadPoolExecutor(max_workers=len(nodes)) as ex:
        eres = list(ex.map(env_check_one, nodes))
    bad = []
    for ip, tag, items in eres:
        print(f"  [{ip}] {tag}  {items}")
        if tag != "READY":
            bad.append(f"{ip}({tag})")
    if bad:
        sys.exit(
            f"✗ 环境未就绪 {len(bad)} 台: {bad}——items 里 FAIL 的项即问题点"
            "（mount=挂载/kuccl_env=环境脚本/torchrun=工具链/kuccl=库可见）"
        )

    # ── 1.5 代码同步（git=各节点走代理自拉 / rsync=master 并发分发含 build 产物）──
    if args.sync:
        t0s = time.time()
        work = [ip for ip in nodes if ip != MASTER]

        if args.sync_mode == "git":
            print(f"── [sync] 各节点 git pull（proxy={args.proxy or '<未设>'}）...")
            proxy_env = ""
            if args.proxy:
                proxy_env = (
                    f"export http_proxy={args.proxy} https_proxy={args.proxy} "
                    f"no_proxy=127.0.0.1,localhost;"
                )
            git_cmd = (
                f"{proxy_env} cd '{repo}' && "
                f"git config --global http.sslVerify false && "
                f"git pull -q 2>&1 | tail -1; true"
            )
            with ThreadPoolExecutor(max_workers=len(work) or 1) as ex:
                outs = list(ex.map(lambda ip: ssh_run(ip, git_cmd, timeout=120), work))
            for ip, (rc, out) in zip(work, outs):
                tail = [ln for ln in out.splitlines() if ln.strip()][-1:] or [
                    "(无输出)"
                ]
                print(f"  [{ip}] {'✓' if rc == 0 else '✗'} {tail[0].strip()}")
            if any(rc != 0 for rc, _ in outs):
                print("  ⚠ 有节点 git 失败——代码可能不一致（建议 --sync-mode rsync）")
            print(
                f"  ✓ git 同步完成（{time.time() - t0s:.0f}s）"
                "——注意：源码变更需在节点重编（build 产物不随 git 传输）"
            )
        else:
            print("── [sync] 代码分发（并发 rsync，含 build 产物）...")
            excludes = "--exclude=.git --exclude=__pycache__ --exclude=logs"

            def push(ip):
                # 新路径首次使用时目标根不存在，rsync 只建末级目录——先 mkdir
                ssh_run(ip, f"mkdir -p '{repo}'", timeout=15)
                t0 = time.time()
                rc = subprocess.run(
                    ["rsync", "-a", "--delete"]
                    + shlex.split(excludes)
                    + [f"{repo}/", f"root@{ip}:{repo}/"],
                    capture_output=True,
                    text=True,
                    timeout=600,
                ).returncode
                print(
                    f"  [sync] {ip} {'✓' if rc == 0 else '✗'} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
                return rc

            with ThreadPoolExecutor(max_workers=len(work) or 1) as ex:
                rcs = list(ex.map(push, work))
            if any(rcs):
                bad = [ip for ip, rc in zip(work, rcs) if rc != 0]
                sys.exit(f"✗ 分发失败节点: {bad}")
            print(f"  ✓ {len(work)} 台分发完成（{time.time() - t0s:.0f}s）")

    if args.gate:
        ok = correctness_gate(args, nodes, repo, conf, tls, args.gate_timeout)
        print(f"\n══ 摘要：{'✓ 通过' if ok else '✗ 未通过'}（--gate 只跑正确性检查）══")
        sys.exit(0 if ok else 1)

    if not args.no_clean:
        print("── [clean] 清理各节点残留进程（超时自动跳过）...")
        # pkill 模式用 [x] 字符类技巧：模式串自身不含被匹配的字面词，
        # 避免 pkill -f 匹配到执行它的 shell 自己（自倾向 = ssh 会话被 -9 杀断）
        # 输出/输入全重定向：被 -9 的残留进程持有的管道不会挂住会话
        # spawn_main：孤儿 multiprocessing worker（曾见 6 个烧满 CPU 一周）
        clean = (
            "pkill -9 -f '[t]orchrun' >/dev/null 2>&1 </dev/null;"
            " pkill -9 -f '[t]orch.distributed.run' >/dev/null 2>&1 </dev/null;"
            " pkill -9 -f '[d]sbench_entry.py' >/dev/null 2>&1 </dev/null;"
            " pkill -9 -f '[t]ests/correctness.py' >/dev/null 2>&1 </dev/null;"
            " pkill -9 -f '[s]pawn_main' >/dev/null 2>&1 </dev/null; true"
        )
        # 清理后复查：残留直接现形，而不是等下轮跑分莫名变慢
        verify = (
            "pgrep -af '[t]orchrun|[d]sbench_entry|[c]orrectness.py|"
            "[s]pawn_main' >/dev/null 2>&1 && echo LEFTOVER || echo CLEAN"
        )

        def clean_one(ip):
            try:
                ssh_run(ip, clean, timeout=30)
                rc, out = ssh_run(ip, verify, timeout=15)
                return ip, ("LEFTOVER" if "LEFTOVER" in out else "OK")
            except subprocess.TimeoutExpired:
                return ip, "TIMEOUT"

        with ThreadPoolExecutor(max_workers=len(nodes)) as ex:
            cres = list(ex.map(clean_one, nodes))
        skipped = [ip for ip, s in cres if s == "TIMEOUT"]
        leftover = [ip for ip, s in cres if s == "LEFTOVER"]
        for ip, s in cres:
            if s == "LEFTOVER":
                print(f"  [clean] {ip} 复查仍有残留进程（模式外的进程，需人工处理）")
            elif s == "TIMEOUT":
                print(f"  [clean] {ip} 清理超时，已跳过")
            else:
                print(f"  [clean] {ip} ✓", flush=True)
        if skipped:
            print(
                f"  [clean] {len(skipped)} 个节点清理超时；若端口仍被占用，"
                "后续运行会明确报错"
            )
        if leftover or skipped:
            print(
                "  [clean] 提示：残留节点会拖慢全组（集合通信锁步），"
                "建议人工 ssh 处理后再跑"
            )
        time.sleep(2)

    backends = [b for b in ("kuccl", "gloo") if b in args.backends]
    if not backends:
        sys.exit(f"✗ --backends 无有效项: {args.backends}")
    port = args.port
    # 端口预检（并行）：rendezvous 端口被残留占用 = worker 侧 900s 假死的头号原因；
    # ss|grep 无匹配时 stdout 为空，但 stderr 横幅会拼进来——必须先滤横幅再判空

    def port_check_one(ip):
        rc, out = ssh_run(
            ip, f"ss -tln | grep -E ':{port}\\b|:{port - 1}\\b' || true", timeout=15
        )
        if "REMOTE HOST IDENTIFICATION HAS CHANGED" in out:
            return ip, "HOSTKEY", ""
        busy = _clean_banner(out).strip()
        return ip, ("BUSY" if busy else "FREE"), busy

    with ThreadPoolExecutor(max_workers=len(nodes)) as ex:
        pres = list(ex.map(port_check_one, nodes))
    for ip, st, busy in pres:
        if st == "HOSTKEY":
            sys.exit(f"✗ [port] {ip} 主机指纹变更——执行 ssh-keygen -R {ip} 后重跑")
        if st == "BUSY":
            sys.exit(
                f"✗ [port] {ip} 的 {port}/{port - 1} 端口已被占用：\n{busy}\n"
                "——先清残留或 --port 换端口段"
            )

    results = {}
    for backend in backends:
        text = launch(backend, args, nodes, repo, conf, port, tls)
        port += 1
        if text is None:
            print(f"  ⚠ {backend} 轮有节点失败——该后端数据可能不全")
        for script, op in OP_SCRIPTS:
            if script in args.ops:
                parsed = parse_raw_output(text or "", script)
                if parsed:
                    results[(backend, op)] = parsed

    all_speedups, pair_lat = [], []
    wins = total = 0
    print("")
    for script, op in OP_SCRIPTS:
        if script not in args.ops:
            continue
        k = results.get(("kuccl", op), {})
        if not k:
            continue
        g = results.get(("gloo", op), {})
        print(f"──── {op} ────")
        if not g:
            # 单后端模式（如算法横评）：无对比判定，仅输出 kuccl 表
            print(f"  {'size':>10s} {'kuccl(us)':>11s}")
            print("  " + "-" * 24)
            for size in sorted(k):
                print(f"  {size:>9d}B {k[size]:>11.1f}")
            continue
        common = sorted(set(k) & set(g))
        if not common:
            print("  无对齐数据（某后端该算子失败？）")
            continue
        print(f"  {'size':>10s} {'kuccl(us)':>11s} {'gloo(us)':>11s} {'延迟提升':>9s}")
        print("  " + "-" * 58)
        for size in common:
            kk, gg = k[size], g[size]
            if kk <= 0 or gg <= 0:
                continue
            up = (gg - kk) / gg * 100
            all_speedups.append(up)
            pair_lat.append((kk, gg))
            total += 1
            wins += 1 if up > 0 else 0
            mark = "★" if up >= args.min_gain else ("✓" if up > 0 else "✗")
            print(f"  {size:>9d}B {kk:>11.1f} {gg:>11.1f} {up:>+8.1f}% {mark}")

    if "gloo" not in backends:
        print("\n（单后端模式：跳过对比判定）")
        sys.exit(0 if RC == 0 else 1)
    if not total:
        sys.exit("✗ 无可对比数据（检查上方各轮错误）")

    avg = sum(all_speedups) / len(all_speedups)
    geo = math.exp(sum(math.log(g / k) for k, g in pair_lat) / len(pair_lat))
    geo_equiv = (1.0 - 1.0 / geo) * 100

    print("\n" + "═" * 72)
    print(f" 判定（平均通信性能提升 ≥{args.min_gain:.0f}%，≥ 者标 ★）")
    print("─" * 72)
    print(f"  逐项胜出 : {wins}/{total} 项 kuccl 快于 gloo")
    print(
        f"  主口径   : 算术平均延迟提升 {avg:+.1f}% → "
        f"{'★' if avg >= args.min_gain else '✗'}"
    )
    print(
        f"  辅口径   : 几何平均加速比 {geo:.2f}x（等效 {geo_equiv:+.1f}%）→ "
        f"{'★' if geo_equiv >= args.min_gain else '✗'}"
    )
    verdict = (avg >= args.min_gain) and (geo_equiv >= args.min_gain)
    print(f" 双口径一致 → {'★' if verdict else '✗'}")
    print("═" * 72)
    sys.exit(0 if (verdict and RC == 0) else 1)


if __name__ == "__main__":
    main()
