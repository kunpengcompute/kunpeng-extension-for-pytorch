"""
kuccl — UCX/UCG-based collective communication backend for PyTorch.

Usage:
    import kuccl                 # auto-registers "kuccl" backend
    torch.distributed.init_process_group(backend="kuccl", ...)

Architecture:
    include/                  — C++ headers (backend.hpp, oob.hpp)
    csrc/                     — C++ implementations
    kuccl/__init__.py         — Python package (library preload + auto-register)

The compiled C++ extension (kuccl_backend) contains an
__attribute__((constructor)) that calls
torch.distributed.Backend.register_backend("kuccl", createBackendKuccl),
so simply importing this package is enough — no manual registration needed.
"""

import ctypes
import importlib
import os
import sys
from pathlib import Path
from typing import Iterable

# ============================================================================
# 1. Locate private HUCX / XUCG runtime
# ============================================================================

_PKG_DIR = Path(__file__).resolve().parent

# Try environment variables first, then fall back to sibling install/ dir
_HUCX_DIR = Path(os.environ.get("HUCX_DIR", _PKG_DIR / "install" / "hucx")).resolve()
_XUCG_DIR = Path(os.environ.get("XUCG_DIR", _PKG_DIR / "install" / "xucg")).resolve()

_HUCX_LIB_DIR = _HUCX_DIR / "lib"
_HUCX_MODULE_DIR = _HUCX_LIB_DIR / "ucx"
_XUCG_LIB_DIR = _XUCG_DIR / "lib"
_XUCG_PLANC_DIR = _XUCG_LIB_DIR / "planc"


def _require_dir(path: Path, name: str) -> None:
    if not path.is_dir():
        raise RuntimeError(f"KuCCL {name} directory does not exist: {path}")


# Validate paths (soft: only warn if using fallback and dirs don't exist yet)
if _HUCX_LIB_DIR.is_dir():
    _require_dir(_HUCX_LIB_DIR, "HUCX lib")
    if _HUCX_MODULE_DIR.is_dir():
        pass  # UCX module dir exists
if _XUCG_LIB_DIR.is_dir():
    _require_dir(_XUCG_LIB_DIR, "XUCG lib")


# ============================================================================
# 2. Set environment before loading any UCX/UCG library
# ============================================================================

os.environ["HUCX_DIR"] = str(_HUCX_DIR)
os.environ["XUCG_DIR"] = str(_XUCG_DIR)
if _HUCX_MODULE_DIR.is_dir():
    os.environ["UCX_MODULE_DIR"] = str(_HUCX_MODULE_DIR)

_force_ucx_tls = os.environ.get("KuCCL_FORCE_UCX_TLS")
if _force_ucx_tls:
    os.environ["UCX_TLS"] = _force_ucx_tls


# ============================================================================
# 3. Check for pre-existing UCX/UCG libraries in the process
# ============================================================================

_UCX_LIBRARY_NAMES = (
    "libucs.so",
    "libucm.so",
    "libuct.so",
    "libucp.so",
    "libucg.so",
    "libuct_ib.so",
    "libucg_planc",
    "libucg_planm",
)


def _get_loaded_ucx_libraries() -> list:
    maps_path = Path(f"/proc/{os.getpid()}/maps")
    if not maps_path.exists():
        return []
    libraries = set()
    try:
        for line in maps_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            fields = line.split()
            if not fields:
                continue
            path = fields[-1]
            if not path.startswith("/"):
                continue
            basename = os.path.basename(path)
            if any(name in basename for name in _UCX_LIBRARY_NAMES):
                libraries.add(os.path.realpath(path))
    except OSError:
        return []
    return sorted(libraries)


_loaded_before = _get_loaded_ucx_libraries()
if _loaded_before:
    print(
        "[KuCCL] WARNING: UCX/UCG libraries already loaded before kuccl init:",
        flush=True,
    )
    for _p in _loaded_before:
        print(f"[KuCCL]   preloaded: {_p}", flush=True)


# ============================================================================
# 4. Preload private HUCX/XUCG with RTLD_GLOBAL (dependency order matters)
# ============================================================================

_RTLD_FLAGS = ctypes.RTLD_GLOBAL | getattr(os, "RTLD_NOW", 0)
_loaded_handles: list = []


def _resolve_library(directory: Path, candidates: Iterable[str]) -> Path:
    searched = []
    for name in candidates:
        path = directory / name
        searched.append(str(path))
        if path.exists():
            return path.absolute()
    raise RuntimeError(
        "KuCCL required library not found. Searched:\n  " + "\n  ".join(searched)
    )


def _load_required_library(
    label: str, directory: Path, candidates: Iterable[str]
) -> ctypes.CDLL:
    path = _resolve_library(directory, candidates)
    try:
        handle = ctypes.CDLL(str(path), mode=_RTLD_FLAGS)
    except OSError as exc:
        raise RuntimeError(f"KuCCL failed to load {label}: {path}: {exc}") from exc
    _loaded_handles.append(handle)
    if os.environ.get("KuCCL_LIBRARY_DEBUG", "0") != "0":
        print(f"[KuCCL] preloaded {label}: {path}", flush=True)
    return handle


# Load order: libucs → libuct → libucp → libucg → libucg_planc_ucx
# (libucs DT_NEEDED libucm; must load from libucs to let ELF resolve the pair)
if _HUCX_LIB_DIR.is_dir():
    _load_required_library("libucs", _HUCX_LIB_DIR, ("libucs.so.0", "libucs.so"))
    _load_required_library("libuct", _HUCX_LIB_DIR, ("libuct.so.0", "libuct.so"))
    _load_required_library("libucp", _HUCX_LIB_DIR, ("libucp.so.0", "libucp.so"))

if _XUCG_LIB_DIR.is_dir():
    _load_required_library("libucg", _XUCG_LIB_DIR, ("libucg.so.0", "libucg.so"))
    if _XUCG_PLANC_DIR.is_dir():
        _load_required_library(
            "libucg_planc_ucx", _XUCG_PLANC_DIR, ("libucg_planc_ucx.so",)
        )

# 库本体 libkuccl（PyInstaller 打包布局：包内 lib/ 目录）。
# 先按 SONAME 登记，后续 kuccl_backend 的 DT_NEEDED libkuccl.so.2 即可解析；
# 普通源码布局下无此目录，自动跳过（走 LD_LIBRARY_PATH / 自愈路径）。
for _so in sorted(_PKG_DIR.glob("lib/libkuccl.so*")):
    _loaded_handles.append(ctypes.CDLL(str(_so), mode=_RTLD_FLAGS))
    break


# ============================================================================
# 5. Import the compiled extension (auto-registers "kuccl" backend)
#
# The extension's __attribute__((constructor)) calls
#   torch.distributed.Backend.register_backend("kuccl", createBackendKuccl)
# so simply importing the module is enough — no manual registration needed.
# ============================================================================

if "kuccl.kuccl_backend" in sys.modules:
    raise RuntimeError(
        "kuccl.kuccl_backend was imported before the private HUCX/XUCG runtime "
        "was initialized. Ensure `import kuccl` comes first."
    )

try:
    import torch  # noqa: F401  (可用性探测)

    _ext = importlib.import_module("kuccl.kuccl_backend")
except ImportError as _e:
    _msg = str(_e)
    # 符号未定义 = .so 与库版本错配，必须原样抛出（不能被回退掩盖）
    if "undefined symbol" in _msg:
        raise RuntimeError(
            f"kuccl.kuccl_backend extension failed to load: {_msg}\n"
            "排查步骤：\n"
            "  1) nm -D ../kuccl/build/libkuccl.so | grep ' T ' | grep kuccl\n"
            "     （应列出 12 个导出接口；缺失 → 重新编译 libkuccl 后重编 kuccl_backend）\n"
            "  2) ldd kuccl/kuccl_backend*.so | grep kuccl  （确认 libkuccl.so 解析到 build 目录的新库）\n"
            "  3) 两者都改过后务必重建 torch_kuccl：cd torch_kuccl && ./build.sh"
        ) from _e
    # 依赖库（libkuccl.so.2 等）不在搜索路径：构建产物就在兄弟目录，
    # 主动 dlopen 绝对路径后重试 import（dlopen 按文件真实 SONAME 登记，
    # 后续 DT_NEEDED 即可解析）。仍失败才抛带排查步骤的错误。
    _saved = _e
    _lib_dir = _PKG_DIR.parent.parent / "kuccl" / "build"
    for _so in sorted(_lib_dir.glob("libkuccl.so*")):
        try:
            _loaded_handles.append(ctypes.CDLL(str(_so), mode=_RTLD_FLAGS))
            break
        except OSError:
            continue
    try:
        _ext = importlib.import_module("kuccl.kuccl_backend")
    except ImportError as _e2:
        raise RuntimeError(
            f"kuccl.kuccl_backend extension failed to load: {_e2}\n"
            f"（已尝试主动加载 {_lib_dir}/libkuccl.so* 仍失败——构建产物缺失？）\n"
            "排查步骤：\n"
            "  1) ls ../kuccl/build/  （确认 libkuccl.so / libkuccl.so.2 存在）\n"
            "  2) 缺失 → 重编：cd ../kuccl && ./build.sh clean && ./build.sh\n"
            "  3) 再重建适配层：cd ../torch_kuccl && ./build.sh"
        ) from _saved


# ============================================================================
# 6. Verify loaded library mappings
# ============================================================================

_loaded_after = _get_loaded_ucx_libraries()
if os.environ.get("KuCCL_LIBRARY_DEBUG", "0") != "0":
    print("[KuCCL] loaded UCX/UCG mappings:", flush=True)
    for _p in _loaded_after:
        print(f"[KuCCL]   {_p}", flush=True)

_private_roots = (str(_HUCX_DIR), str(_XUCG_DIR))
_unexpected = [p for p in _loaded_after if not p.startswith(_private_roots)]
if _unexpected:
    print(
        "[KuCCL] WARNING: UCX/UCG libraries outside private runtime loaded:", flush=True
    )
    for _p in _unexpected:
        print(f"[KuCCL]   unexpected: {_p}", flush=True)


# ============================================================================
# 7. Public API
# ============================================================================

# Re-export the factory for manual use if needed
create_kuccl_backend = _ext.create_kuccl_backend

__version__ = "2.6.0"
__all__ = ["create_kuccl_backend"]
