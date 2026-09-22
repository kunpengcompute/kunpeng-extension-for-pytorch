#!/usr/bin/env python3
"""
torch_kuccl — PyTorch c10d 适配层

独立构建的 Python 扩展，链接 libkuccl.so（不重编库代码）。
适配层只链接 libkuccl.so，不含任何库代码。

前置：
  1. 先构建 kuccl 库（兄弟目录）：
      cd ../kuccl && ./build.sh lib
  2. 环境变量：
      export HUCX_DIR=/path/to/hucx
      export XUCG_DIR=/path/to/xucg

构建：
  cd torch_kuccl
  python setup.py build_ext --inplace

产物：kuccl/kuccl_backend.so（包内；import kuccl 即自动注册 "kuccl" backend）
"""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE      = Path(__file__).parent.resolve()
LIBKuCCL  = (HERE.parent / "kuccl").resolve()        # 库本体（兄弟目录）
HUCX_DIR = os.environ.get("HUCX_DIR")
XUCG_DIR = os.environ.get("XUCG_DIR")

if not HUCX_DIR:
    raise SystemExit("请先 export HUCX_DIR=/path/to/hucx")
if not XUCG_DIR:
    raise SystemExit("请先 export XUCG_DIR=/path/to/xucg")

ext_module = CppExtension(
    name="kuccl.kuccl_backend",
    sources=[
        "csrc/backend.cpp",
        "csrc/oob.cpp",                # Store 版 OOB 工具（适配层私有）
    ],
    include_dirs=[
        str(HERE / "include"),          # backend.hpp / oob.hpp（本包）
        str(LIBKuCCL / "include"),      # kuccl.h / types.hpp（库公开头）
        os.path.join(HUCX_DIR, "include"),
        os.path.join(XUCG_DIR, "include"),
    ],
    library_dirs=[
        str(LIBKuCCL / "build"),        # libkuccl.so 的位置
        os.path.join(HUCX_DIR, "lib"),
        os.path.join(XUCG_DIR, "lib"),
        os.path.join(XUCG_DIR, "lib", "planc"),
    ],
    runtime_library_dirs=[
        str(LIBKuCCL / "build"),
        os.path.join(HUCX_DIR, "lib"),
        os.path.join(XUCG_DIR, "lib"),
        os.path.join(XUCG_DIR, "lib", "planc"),
    ],
    libraries=["kuccl",                 # 链接 libkuccl.so（不重编库）
               "ucp", "uct", "ucs", "ucm", "ucg"],
    extra_compile_args=["-O2", "-std=c++17", "-fvisibility=hidden"],
    extra_link_args=["-Wl,--as-needed"],
)

setup(
    name="kuccl",
    version="2.6.0",
    description="KuCCL PyTorch backend adapter (links libkuccl.so)",
    packages=["kuccl"],
    package_dir={"kuccl": "kuccl"},
    ext_modules=[ext_module],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.12",
)
