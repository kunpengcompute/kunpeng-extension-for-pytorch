"""config/default.py —— 920F 现网 16 节点 RoCE 集群（master = 29.204.23.30）

装载方式（compare_nodes.py 自动 import；--conf 换档）：
    python benchmarks/communication/compare_nodes.py --conf default
换集群：复制本文件为 config/<集群名>.py 改字段，--conf <集群名>。
"""

# ── 节点 ──────────────────────────────────────────────────────
# 节点清单（支持紧凑写法 "前缀|起-止,起-止"；普通 IP 原样；主流程取前 nnodes 台）
NODES = [
    "29.204.23.|30-37,40-47",
]
# master 覆写（None = 取清单第 1 台）；master 承担 TCPStore / CSV / 日志汇总
MASTER_NODE = None

# ── 代码 ──────────────────────────────────────────────────────
REPO_DIR = "/root/pacific_ext/w30084706/kuccl"          # 各节点上的仓库路径
REPO_URL = "https://atomgit.com/W-ayang/test.git"       # git 同步模式的仓库地址

# ── 传输 ──────────────────────────────────────────────────────
TLS_SINGLE = "sm,self"     # 单机默认 UCX_TLS（需 hucx 构建启用 sm 组件）
TLS_MULTI = "rc,sm,self"   # 跨节点：rc 走网间；sm 供同节点对使用（na_rolling
                           # 等 node-aware 算法的楼内阶段依赖 shm，纯 rc 时
                           # 楼内流量绕道网卡环回，实测 all_gather 完败）

# ── 诊断 ──────────────────────────────────────────────────────
RETRANS_NICS = ["roh0", "roh1", "roh2", "roh3",
                "roh4", "roh5", "roh6", "roh7"]   # ethtool -S 重传计数网卡

# ── sync（开发机 → master）────────────────────────────────────
SYNC_SRC = "/root/pacific_ext/w30084706/kuccl"      # 开发机上仓库路径（信息档）
