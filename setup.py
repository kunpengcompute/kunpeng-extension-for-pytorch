#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# KPEX is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#        http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#

from setuptools import setup, find_packages
from torch.utils import cpp_extension
import glob
from pathlib import Path
import os

root = Path(__file__).parent

sources = [
    "csrc/*.cpp",
    "csrc/utils/*.cpp",
    "csrc/aten/*.cpp",
    "csrc/comm/*.cpp",
    "csrc/comm/local/*.cpp",
    "csrc/tpp/alphafold/*.cpp",
    "csrc/kudnn/*.cpp",
    "csrc/cpu/aten/*.cpp"
]

sources_example = [
    "csrc/utils/*.cpp",
    "examples/csrc/*.cpp",
    "examples/csrc/kupl_example/*.cpp",
]

sources = [j for i in sources for j in glob.glob(i)]
sources_example = [j for i in sources_example for j in glob.glob(i)]
extra_compile_args = []
extra_link_args = []
include_dirs = [(root / "csrc").as_posix()]
include_dirs_example = [(root / "csrc").as_posix()]
include_dirs_example += [(root / "examples/csrc").as_posix()]

extra_compile_args += ["-fopenmp"]
if os.environ["KPEX_BUILD_TYPE"] == "release":
    extra_compile_args += ["-O3"]
elif os.environ["KPEX_BUILD_TYPE"] == "debug":
    extra_compile_args += ["-O0", "-g"]
else:
    print("requires env KPEX_BUILD_TYPE (release / debug)")

library_dirs = []
libraries = []
KUTACC_ROOT = os.environ.get("KUTACC_ROOT", None)
if KUTACC_ROOT:
    include_dirs += [f"{KUTACC_ROOT}/include"]
    library_dirs = [f"{KUTACC_ROOT}/lib"]
    libraries += ["kutacc"]

KUDNN_ROOT = os.environ.get("KUDNN_ROOT", None)
if KUDNN_ROOT:
    include_dirs += [f"{KUDNN_ROOT}/include"]
    library_dirs += [f"{KUDNN_ROOT}/lib"]
    libraries += ["kudnn"]

library_dirs_example = []
libraries_example = []
KUPL_ROOT = os.environ.get("KUPL_ROOT", None)
if KUPL_ROOT:
    library_dirs += [f"{KUPL_ROOT}/include"]
    libraries += [f"{KUPL_ROOT}/lib"]
    libraries += ["kupl"]

    library_dirs_example += [f"{KUPL_ROOT}/include"]
    libraries_example += [f"{KUPL_ROOT}/lib"]
    libraries_example = ["kupl"]

security_compile_args = [
    "-fPIC",
    "-fstack-protector-strong",
    "-D_FORTIFY_SOURCE=2",
    "-ftrapv",
    "-fvisibility=hidden",
]
security_link_args = [
    "-Wl,-z,relro,-z,now",
    "-Wl,-z,noexecstack",
    "-s",
]

extra_compile_args += security_compile_args
extra_link_args += security_link_args

setup(
    name="kunpeng-pytorch-extension",
    version="0.0.1",
    author="KPEX",
    packages=find_packages(),
    ext_modules=[
        cpp_extension.CppExtension(
            name="kpex._C",
            sources=sources,
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            extra_compile_args={"cxx": extra_compile_args},
            extra_link_args = extra_link_args,
        ),

        cpp_extension.CppExtension(
            name="kpex._C_example",
            sources=sources_example,
            include_dirs=include_dirs_example,
            libraries=libraries_example,
            library_dirs=libraries_example,
            extra_compile_args={"cxx": extra_compile_args},
            extra_link_args = extra_link_args,
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension}
)
