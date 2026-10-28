"""config/example.py —— 新集群配置模板

复制为 config/<你的集群名>.py 改字段，使用：
    python benchmarks/communication/compare_nodes.py --conf <你的集群名>
换集群零改代码：只加配置档。
"""

# ── 节点 ──────────────────────────────────────────────────────
# 支持紧凑写法 "前缀|起-止,起-止,单值"（空格容忍），普通 IP 原样；两种可混用
NODES = ["29.204.23.|50-57,60-67", "10.0.0.1"]
MASTER_NODE = None  # None = 取清单第 1 台

# ── 代码 ──────────────────────────────────────────────────────
REPO_DIR = "/root/kuccl/kuccl"  # 各节点上的仓库路径（同步目标）
REPO_URL = (
    "https://example.com/your/repo.git"  # git 同步模式的仓库地址（--sync-mode git）
)

# ── 传输 ──────────────────────────────────────────────────────
TLS_SINGLE = "sm,self"  # 常见取值：rc / rc,self / ud,self / sm,self / tcp
TLS_MULTI = "rc,sm,self"  # 跨节点推荐：rc 网间 + sm 楼内（node-aware 算法依赖）

# ── 诊断 ──────────────────────────────────────────────────────
RETRANS_NICS = ["eth0"]  # ip link 看接口名；每次跑分前后自动快照进日志

# ── sync（开发机 → master）────────────────────────────────────
SYNC_SRC = "/root/kuccl/kuccl"
