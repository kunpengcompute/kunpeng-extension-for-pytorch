#!/usr/bin/env python3
"""Standalone local CPU test launcher; see PYTORCH_CPU_TESTS.md.

Do not wrap in torchrun: this parent supervises every rank itself. The parent
imports no torch/KuCCL, and failures never silently select another backend.
"""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import traceback
import unittest

from pytorch_cpu_suite import make_cases, memory_estimate, run_case


HERE = Path(__file__).resolve().parent
PROFILES = ("kupl-default", "kupl-doorbell", "kupl-hier", "xucg", "gloo")


def profile_environment(profile, threads):
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    if profile == "gloo":
        return {}
    result = {"KUCCL_BACKEND_TYPE": "xucg" if profile == "xucg" else "kupl"}
    if profile.startswith("kupl-"):
        algo = "hier" if profile == "kupl-hier" else "linear"
        result.update(
            KUCCL_KUPL_AR_ALGO="hier" if profile == "kupl-hier" else "flat",
            KUCCL_KUPL_AG_ALGO=algo,
            KUCCL_KUPL_AG_THREADS=str(threads),
            KUCCL_KUPL_AG_SYNC="doorbell" if profile == "kupl-doorbell" else "fence",
            KUCCL_KUPL_A2A_ALGO="direct",
            KUCCL_KUPL_BC_ALGO="direct",
        )
    return result


def merge_rank_results(records, world_size):
    ranks = [record.get("rank") for record in records]
    valid = len(ranks) == world_size and set(ranks) == set(range(world_size))
    statuses = {record.get("status") for record in records}
    if valid and statuses == {"PASS"}:
        status = "PASS"
    elif valid and statuses == {"SKIP"}:
        status = "SKIP"
    else:
        status = "FAIL"
    return {
        "status": status,
        "ranks": records,
        "error": "" if valid else "missing, duplicate, or invalid rank results",
    }


def report_passed(records):
    statuses = [record.get("status") for record in records]
    return (
        bool(statuses)
        and "PASS" in statuses
        and all(s in ("PASS", "SKIP") for s in statuses)
    )


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_progress(path):
    # Some WSL/Windows shared mounts expose a brief absence during replace().
    # Retry on the next watchdog tick; only a valid changed index resets time.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value["index"]
    except (FileNotFoundError, PermissionError, json.JSONDecodeError, KeyError):
        return None


def bind_rank(args, rank):
    if not args.numa_nodes:
        return {
            "numa_node": None,
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
        }
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("--numa-nodes requires Linux sched_setaffinity")
    node = args.numa_nodes[rank % len(args.numa_nodes)]
    text = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
    cpus = set()
    for item in text.split(","):
        bounds = [int(i) for i in item.split("-")]
        cpus.update(range(bounds[0], bounds[-1] + 1))
    cpus &= os.sched_getaffinity(0)
    if not cpus:
        raise RuntimeError(f"NUMA node {node} has no CPUs in the inherited affinity")
    os.sched_setaffinity(0, cpus)
    return {"numa_node": node, "cpu_affinity": sorted(os.sched_getaffinity(0))}


def loaded_communication_libraries():
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return []
    paths = set()
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        path = line.split()[-1]
        if path.startswith("/") and any(
            name in Path(path).name
            for name in ("kuccl", "libkupl", "libucg", "libucp", "libucs")
        ):
            paths.add(path)
    return sorted(paths)


def rank_worker(args):
    # This function runs only in a child process with explicit rank environment.
    rank, world_size = int(os.environ["RANK"]), args.world_sizes[0]
    profile = args.profiles[0]
    run_dir = Path(args.worker_dir)
    placement = bind_rank(args, rank)
    import torch
    import torch.distributed as dist

    backend = "gloo" if profile == "gloo" else "kuccl"
    package = None
    if backend == "kuccl":
        # Prefer the checkout's adapter; its compiled extension must be installed/built.
        sys.path.insert(0, str(HERE.parents[2] / "csrc" / "torch_kuccl"))
        import kuccl

        package = str(Path(kuccl.__file__).resolve())
        if not dist.is_backend_available("kuccl"):
            raise RuntimeError(
                "import kuccl did not register the kuccl distributed backend"
            )

    timeout = timedelta(seconds=args.timeout)
    store = dist.TCPStore(
        os.environ["MASTER_ADDR"],
        int(os.environ["MASTER_PORT"]),
        world_size,
        rank == 0,
        timeout=timeout,
        wait_for_workers=True,
    )
    dist.init_process_group(
        backend,
        store=dist.PrefixStore("collective/", store),
        rank=rank,
        world_size=world_size,
        timeout=timeout,
    )
    metadata = dict(
        rank=rank,
        torch_version=torch.__version__,
        torch_git_version=torch.version.git_version,
        backend=backend,
        kuccl_package=package,
        placement=placement,
        loaded_communication_libraries=loaded_communication_libraries(),
        all_gather_single_alias=not hasattr(dist, "all_gather_single"),
        environment={
            key: value
            for key, value in os.environ.items()
            if key.startswith(("KUCCL_", "KUPL_", "TORCH_KUCCL_", "UCX_"))
        },
        hucx_dir=os.environ.get("HUCX_DIR"),
        xucg_dir=os.environ.get("XUCG_DIR"),
    )
    write_json(run_dir / f"rank-{rank}.metadata.json", metadata)
    cases = make_cases(args.suite, world_size, args.repeats)
    if args.case_filter:
        cases = [case for case in cases if args.case_filter in case.id]
    budget = args.max_memory_mib * 1024 * 1024
    for index, case in enumerate(cases):
        if rank == 0:
            write_json(run_dir / "progress.json", {"index": index, "case": case.id})
        begin = time.monotonic()
        result = {"rank": rank, "status": "PASS"}
        try:
            if memory_estimate(case, world_size) > budget:
                raise unittest.SkipTest(
                    "estimated tensor memory exceeds --max-memory-mib per rank"
                )
            run_case(case, rank, world_size, profile)
        except unittest.SkipTest as error:
            result.update(status="SKIP", error=str(error))
        except Exception:
            result.update(status="FAIL", error=traceback.format_exc())
            print(f"FAIL {case.id}\n{result['error']}", flush=True)
        result["seconds"] = time.monotonic() - begin
        key = f"verdict/{index}/{rank}"
        store.set(key, json.dumps(result))
        keys = [f"verdict/{index}/{peer}" for peer in range(world_size)]
        store.wait(keys, timeout)
        records = [json.loads(store.get(key).decode()) for key in keys]
        if rank == 0:
            merged = dict(merge_rank_results(records, world_size), case=case.to_dict())
            with (run_dir / "cases.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(merged, ensure_ascii=False) + "\n")
            print(f"{index + 1}/{len(cases)} {merged['status']} {case.id}", flush=True)
        # The OOB release ensures rank zero persisted the verdict before the next case.
        if rank == 0:
            store.set(f"release/{index}", "1")
        store.get(f"release/{index}")
    dist.destroy_process_group()
    store.set(f"done/{rank}", "1")
    if rank == 0:
        store.wait([f"done/{peer}" for peer in range(world_size)], timeout)
        store.set("shutdown", "1")
    store.get("shutdown")
    store.set(f"ack/{rank}", "1")
    if rank == 0:
        store.wait([f"ack/{peer}" for peer in range(world_size)], timeout)
    return 0


def read_records(run_dir):
    path = run_dir / "cases.jsonl"
    if not path.exists():
        return [], None
    records = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                not isinstance(record, dict)
                or record.get("status") not in ("PASS", "FAIL", "SKIP")
                or not isinstance(record.get("case"), dict)
                or "id" not in record["case"]
            ):
                raise ValueError(f"invalid record schema on line {number}")
            records.append(record)
    except (OSError, UnicodeError, ValueError) as error:
        return records, f"cannot read complete case log: {error}"
    return records, None


def observed_routes(run_dir, world_size):
    routes = {"allreduce": set(), "allgather": set()}
    path = run_dir / "rank-0.log"
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        for tag, operation in (("AR", "allreduce"), ("AG", "allgather")):
            for size, route in re.findall(
                r"\[" + tag + r"-route\][^\n]*?ws=(\d+)\s+[^\n]*?→\s+(\S+)", text
            ):
                if int(size) == world_size:
                    routes[operation].add(route)
    return {operation: sorted(values) for operation, values in routes.items()}


def supervise(args, profile, world_size, threads, run_dir):
    run_dir.mkdir(parents=True)
    cases = make_cases(args.suite, world_size, args.repeats)
    if args.case_filter:
        cases = [case for case in cases if args.case_filter in case.id]
    env = os.environ.copy()
    # Remove inherited algorithm knobs, then record/use the requested profile.
    for key in list(env):
        if key.startswith("KUCCL_KUPL_"):
            del env[key]
    env.update(profile_environment(profile, threads))
    if profile.startswith("kupl-"):
        env["KUCCL_KUPL_A2A_ALGO"] = args.a2a_algo
        env["KUCCL_KUPL_BC_ALGO"] = args.bc_algo
    env["TORCH_KUCCL_ALLREDUCE_INPLACE"] = str(args.ar_inplace)
    env.update(
        MASTER_ADDR="127.0.0.1",
        WORLD_SIZE=str(world_size),
        LOCAL_WORLD_SIZE=str(world_size),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS=str(args.torch_threads),
        MKL_NUM_THREADS=str(args.torch_threads),
    )
    if profile != "gloo":
        env.update(KUCCL_LOG_LEVEL="TRACE", KUCCL_LOG_RANKS="0")
        env.pop(
            "KUCCL_LOG_FILE", None
        )  # Keep route evidence in the supervised rank log.
    # A loopback ephemeral port is released just before spawning. Bind failures
    # remain hard failures in rank logs; never attach to an unrelated server.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        env["MASTER_PORT"] = str(sock.getsockname()[1])
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-dir",
        str(run_dir),
        "--suite",
        args.suite,
        "--profiles",
        profile,
        "--world-sizes",
        str(world_size),
        "--threads",
        str(threads),
        "--timeout",
        str(args.timeout),
        "--max-memory-mib",
        str(args.max_memory_mib),
        "--repeats",
        str(args.repeats),
    ]
    if args.case_filter:
        command += ["--case-filter", args.case_filter]
    if args.numa_nodes:
        command += ["--numa-nodes", *map(str, args.numa_nodes)]
    processes, logs = [], []
    fatal = None
    last_progress, last_change = None, time.monotonic()
    try:
        for rank in range(world_size):
            output = (run_dir / f"rank-{rank}.log").open("w", encoding="utf-8")
            logs.append(output)
            rank_env = dict(env, RANK=str(rank), LOCAL_RANK=str(rank))
            processes.append(
                subprocess.Popen(
                    command, env=rank_env, stdout=output, stderr=subprocess.STDOUT
                )
            )
        while True:
            codes = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in codes):
                fatal = f"rank process failed; exit codes {codes}"
                break
            if all(code is not None for code in codes):
                break
            progress = read_progress(run_dir / "progress.json")
            if progress is not None and progress != last_progress:
                last_progress, last_change = progress, time.monotonic()
            if time.monotonic() - last_change > args.timeout + 15:
                fatal = f"watchdog: no case progress for {args.timeout + 15}s; last={last_progress}"
                break
            time.sleep(0.1)
    except Exception:
        fatal = traceback.format_exc()
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.wait()
        for output in logs:
            output.close()
    records, read_error = read_records(run_dir)
    fatal = fatal or read_error
    completed = [record.get("case", {}).get("id") for record in records]
    expected = [case.id for case in cases]
    if completed != expected[: len(completed)] or len(completed) > len(expected):
        fatal = "recorded case order/count differs from the planned matrix"
    if len(completed) < len(expected):
        for offset, case in enumerate(cases[len(completed) :]):
            records.append(
                {
                    "case": case.to_dict(),
                    "status": "FAIL" if offset == 0 else "NOT_RUN",
                    "error": fatal or "worker exited without reporting every case",
                }
            )
    routes = observed_routes(run_dir, world_size)
    hierarchical = any("hier" in route for route in routes["allreduce"]) and any(
        "hier" in route for route in routes["allgather"]
    )
    coverage = None
    if profile == "kupl-hier":
        coverage = dict(
            hierarchical_routes_observed=hierarchical,
            note="Only TRACE route evidence verifies hierarchical execution; profile name alone does not.",
        )
        if not hierarchical:
            print(
                f"COVERAGE: kupl-hier ws={world_size}: hierarchical AR+AG routes unverified; inspect rank-0.log",
                flush=True,
            )
            if args.require_hier:
                fatal = (
                    fatal
                    or "--require-hier: hierarchical allreduce and allgather routes not observed"
                )
    if fatal:
        records.append(
            {
                "case": {"id": "runner/process-lifecycle"},
                "status": "FAIL",
                "error": fatal,
            }
        )
    result = dict(
        profile=profile,
        world_size=world_size,
        threads=threads,
        passed=report_passed(records),
        counts=dict(Counter(r["status"] for r in records)),
        environment={
            k: v
            for k, v in env.items()
            if k.startswith(("KUCCL_", "KUPL_", "TORCH_KUCCL_", "UCX_"))
        },
        logs=str(run_dir),
        cases=records,
        fatal=fatal,
    )
    result.update(observed_routes=routes, hierarchical_coverage=coverage)
    write_json(run_dir / "report.json", result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--profiles", choices=PROFILES, nargs="+", default=["kupl-default"]
    )
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[2])
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[4],
        help="KuPL copy/reduce worker counts",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="OMP/MKL threads, separate from KuPL workers",
    )
    parser.add_argument("--a2a-algo", choices=("direct", "native"), default="direct")
    parser.add_argument("--bc-algo", choices=("direct", "native"), default="direct")
    parser.add_argument("--ar-inplace", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--require-hier",
        action="store_true",
        help="fail kupl-hier runs unless TRACE proves hierarchical AR and AG",
    )
    parser.add_argument(
        "--numa-nodes",
        type=int,
        nargs="+",
        help="rank round-robin node binding; allow all permitted CPUs on node",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="per-case/init seconds; parent watchdog adds 15s",
    )
    parser.add_argument(
        "--max-memory-mib",
        type=int,
        default=128,
        help="estimated peak tensor memory per rank; excludes backend scratch",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case-filter", default="", help="substring of stable case ID")
    parser.add_argument("--output", type=Path, default=HERE / "pytorch_cpu_results")
    parser.add_argument(
        "--list",
        action="store_true",
        help="print planned cases without importing torch",
    )
    parser.add_argument("--worker-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (
        any(size < 1 or size > 16 for size in args.world_sizes)
        or any(thread < 1 or thread > 64 for thread in args.threads)
        or min(args.timeout, args.max_memory_mib, args.repeats, args.torch_threads) < 1
    ):
        parser.error(
            "world sizes must be 1..16, KuPL threads 1..64; other numeric options must be positive"
        )
    for size in args.world_sizes:
        cases = make_cases(args.suite, size, args.repeats)
        if args.case_filter and not any(args.case_filter in case.id for case in cases):
            parser.error(f"case filter selects no tests at world size {size}")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker_dir:
        return rank_worker(args)
    if args.list:
        for size in args.world_sizes:
            for case in make_cases(args.suite, size, args.repeats):
                if not args.case_filter or args.case_filter in case.id:
                    print(
                        json.dumps(
                            dict(
                                case.to_dict(),
                                world_size=size,
                                estimated_bytes=memory_estimate(case, size),
                            )
                        )
                    )
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output.resolve() / stamp
    output.mkdir(parents=True)
    upstream = json.loads(
        (HERE / "pytorch_cpu_upstream.json").read_text(encoding="utf-8")
    )
    try:
        checkout_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        checkout_revision = None
    write_json(
        output / "manifest.json",
        dict(
            arguments={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            python=sys.version,
            upstream=upstream,
            checkout_revision=checkout_revision,
        ),
    )
    results = []
    print(f"Results: {output}", flush=True)
    for profile in dict.fromkeys(args.profiles):
        for size in dict.fromkeys(args.world_sizes):
            thread_counts = args.threads if profile.startswith("kupl-") else [1]
            for threads in dict.fromkeys(thread_counts):
                name = f"{profile}-ws{size}-threads{threads}"
                print(f"RUN {name}", flush=True)
                result = supervise(args, profile, size, threads, output / name)
                results.append(result)
                print(
                    f"{'PASS' if result['passed'] else 'FAIL'} {name}: {result['counts']}",
                    flush=True,
                )
                write_json(output / "summary.json", results)
    passed = bool(results) and all(result["passed"] for result in results)
    print(
        f"{'PASS' if passed else 'FAIL'}: report {output / 'summary.json'}", flush=True
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
