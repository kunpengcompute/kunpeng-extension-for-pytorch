# KPEX

## 🔥Release Notes

- [2026/03] The Kunpeng Extension for PyTorch (KPEX) project went live, introducing AlphaFold2 fused operator optimization and seamless backend support for KuDNN and KUPL.

## 🚀Overview

KPEX is an extension built for the Kunpeng platform to enhance specific operator performance and achieve superior execution efficiency.

## 📝Version Mapping

- Operating platform
    - Kunpeng 920 Pro
- System specifications
    - openEuler 22.03 LTS SP4 AArch64

## ⚡️Build and Installation

Follow the build and installation guide below for a quick, start-from-scratch experience of the project.

### 1. Set up the environment.

Before building and installing KPEX, PyTorch must be installed from source. You are advised to use an environment management tool to isolate and manage dependencies. Miniconda is recommended for [download](https://repo.anaconda.com/miniconda) and [installation](https://www.anaconda.com/docs/getting-started/miniconda/install/overview).

### 2. [Obtain the HPCKit software package](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_007.html).

> https://www.hikunpeng.com/developer/hpc/hpckit-download

### 3. [Install HPCKit](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_012.html).

#### Extract the HPCKit software package (replace the version number in the example with your actual version).

```shell
tar xvf HPCKit_26.0.RC1_Linux-aarch64.tar.gz
```

#### Install HPCKit.

```shell
sh HPCKit_26.0.RC1_Linux-aarch64/install.sh -y --prefix=[HPCKit_installation_directory]
```

### 4. [Set environment variables](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_014.html).

#### Install the Module environment management tool (requires yum repository to be configured).
```
yum install environment-modules
```

#### Load the module.

```
module use [HPCKit_installation_directory]/HPCKit/latest/modulefiles
```

#### Load the environment variables of the compiler (replace the version number in the example with your actual version).

```
module load bisheng/compiler5.1.0.2/bishengmodule
```

#### Load the environment variables of dependencies.

```
module load bisheng/kutacc26.0.RC1/kutacc
module load bisheng/kupl26.0.RC1/release
module load bisheng/hmpi26.0.RC1/release
module load bisheng/kudnn26.0.RC1/kudnn
module load bisheng/kml26.0.RC1/kblas/multi
```

### 5. Build and install from source.

Before compiling and installing KPEX, you must first complete the [compilation and installation of PyTorch](requirements_install.md)

- If you choose to download the source code package
#### Upload and extract the KPEX project source package (replace the branch name in the example with your actual branch).
```
unzip kunpeng-extension-for-pytorch-main.zip
```

#### Navigate to the root directory of the KPEX project source code (replace the branch name in the example with your actual branch).
```
cd kunpeng-extension-for-pytorch-main
```

#### Build and install.
Run the following command to build and install:

```
CFLAGS="-stdlib=libstdc++ -lstdc++" KPEX_BUILD_TYPE=release HMPI_ROOT=<path_to_HPCKit>/HPCKit/latest/hmpi/bisheng/release/hmpi KUTACC_ROOT=<path_to_HPCKit>/HPCKit/latest/kutacc/bisheng  KUPL_ROOT=<path_to_HPCKit>/HPCKit/latest/kupl/bisheng/release KUDNN_ROOT=<path_to_HPCKit>/HPCKit/latest/kudnn/bisheng pip install --root-user-action=ignore --no-build-isolation --editable .
```

> **Note**: Replace <path_to_HPCKit> with the actual installation directory of HPCKit.

### 6. Verify the installation.

Run the `python` command to access the Python CLI and run the following command:

```
import torch
```

If no error message is displayed, the build and installation were successful.

## 📖Tutorials

If you are familiar with the **build and installation process** and would like to **gain a deeper understanding of the project**, please visit the following detailed tutorials:

[Developer Guide](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_123.html): Master API functionality and development with in-depth instructions tailored for developers of all levels.

## 🔍Directory Structure

The project directory structure is as follows:

```
├── csrc                           # Project C++ source directory
│   ├── kudnn                      # KuDNN operator backend
│   ├── tpp                        # Compute fused operator
│   ├── utils                      # Common utilities
│   └── kpex.cpp                   # C++ APIs for Python integration
├── kpex                           # Project Python APIs
│   ├── tpp                        # Compute fused operator
│   ├── __init__.py
│   └── frontend.py                # External APIs
├── kpex_example                   # Project examples
│   ├── csrc                       # C++ code directory for project examples
│   ├── kupl_example               # Python code directory for backend KUPL examples
│   ├── __init__.py
│   └── optimization_test.py       # Operator optimization test example
├── thirdparty                     # PyTorch compatibility patch
├── LICENSE
├── README.md
└── setup.py                       # Project build script
```

## 🤝Contact Us

Features and documentation are updated regularly. Please follow the latest version for the most up-to-date information.

- **Issue feedback**: Submit queries or report bugs via [Issues](https://atomgit.com/kunpengcompute/kunpeng-extension-for-pytorch/issues).
- **Community interaction**: Join discussions and share ideas via [Discussions](https://atomgit.com/kunpengcompute/kunpeng-extension-for-pytorch/discussions).
- **Technical columns**: Access in-depth technical articles, serialized tutorials, and best practices through the [Kunpeng Community](https://www.hikunpeng.com/developer/techArticles).
