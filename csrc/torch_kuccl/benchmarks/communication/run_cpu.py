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

import torch.distributed as distributed_c10d


def _worker_env(i, world_size, master_addr, master_port):
    env = os.environ.copy()
    env['RANK'] = str(i)
    env['WORLD_SIZE'] = str(world_size)
    env['LOCAL_RANK'] = str(i)
    env['MASTER_ADDR'] = master_addr
    env['MASTER_PORT'] = master_port
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--world-size', type=int, default=2, help='number of ranks to spawn')
    ap.add_argument('--master-addr', type=str, default='127.0.0.1')
    ap.add_argument('--master-port', type=str, default='29500')
    args, rest = ap.parse_known_args()

    if rest and rest[0] == '--':
        rest = rest[1:]

    if args.world_size < 2:
        # a single process would measure zero/trivial comm cost
        print(f'ERROR: --world-size must be >= 2 to measure communication, got {args.world_size}')
        sys.exit(1)

    if not rest:
        print('ERROR: pass a benchmark script after "--", e.g. "-- all_reduce.py --scan --maxsize 16"')
        sys.exit(1)

    script = rest[0]
    passthrough = rest[1:]
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), script))
    if not os.path.exists(script_path):
        print(f'ERROR: benchmark script not found: {script_path}')
        sys.exit(1)

    procs = []
    for i in range(args.world_size):
        cmd = [sys.executable, script_path, '--local_rank', str(i)] + passthrough
        env = _worker_env(i, args.world_size, args.master_addr, args.master_port)
        procs.append(subprocess.Popen(cmd, env=env))

    ret = 0
    for p in procs:
        rc = p.wait()
        if rc != 0:
            ret = rc
    sys.exit(ret)


if __name__ == '__main__':
    main()