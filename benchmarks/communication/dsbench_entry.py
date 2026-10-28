#!/usr/bin/env python3
"""dsbench_entry.py —— DeepSpeed communication 套件的多节点 torchrun 入口

零侵入：不修改 communication/ 套件任何文件（保持上游口径权威），本入口
只做入口适配：
  1. 剥离 torchrun 兼容模式可能注入的 --local-rank（连字符）——套件
     parser 只认 --local_rank
  2. 从 LOCAL_RANK 环境变量补 local_rank
  3. 调 run_all.main()（backend=kuccl 时套件的 ensure_custom_backend
     会 import kuccl 完成注册）

用法（多节点：每节点一条，node_rank 各异；单机：去掉 --nnodes 等）：
  torchrun --nnodes=8 --nproc_per_node=16 --node_rank=$R \
      --master_addr=<master-ip> --master_port=29510 \
      dsbench_entry.py --backend kuccl --dist torch --scan --maxsize 21 \
      --trials 50 --warmups 5 --raw \
      --all-reduce --all-gather --all-to-all --broadcast

  pt2pt 默认不含（kuccl 无点对点接口）；需要时自行加 --pt2pt。
"""

import os
import sys

# UCG/UCX 日志是进程级开关且无 rank 过滤——128 个进程各打一份等于刷屏。
# 只保留全局 rank 0 一份：非 0 进程在加载任何 UCX/UCG 之前剥掉开关
for _log_var in ("UCG_LOG_LEVEL", "UCX_LOG_LEVEL"):
    if os.environ.get(_log_var) and os.environ.get("RANK", "0") != "0":
        del os.environ[_log_var]

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../"))
sys.path.insert(0, HERE)

# 剥离 torchrun 旧兼容注入的 --local-rank / --local-rank=N
argv = [
    a for a in sys.argv[1:] if a != "--local-rank" and not a.startswith("--local-rank=")
]
sys.argv = [sys.argv[0]] + argv

from communication.utils import benchmark_parser  # noqa: E402
from communication import run_all  # noqa: E402

args = benchmark_parser().parse_args()
if args.local_rank is None:
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
run_all.main(args, args.local_rank)
