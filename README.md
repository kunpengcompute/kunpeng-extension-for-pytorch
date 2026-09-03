# Kunpeng Extension for PyTorch

## 🔥Release Notes

- [2026/03] KPEX项目首次上线，支持AlphaFold2应用计算融合算子优化、KuDNN算子后端对接与KUPL后端对接。

## 🚀概述

Kunpeng Extension for PyTorch（以下简称KPEX）是为鲲鹏平台打造的用于提升部分算子性能的PyTorch扩展，实现在鲲鹏平台上获得性能提升。

## 📝版本配套

- 运行平台
    - 鲲鹏 920 专业版
- 系统规格
    - openEuler 22.03（LTS-SP4）AArch64

## ⚡️编译安装

若您希望**从零到一快速体验**项目能力，请参照下述编译安装教程。

#### 1. 环境准备

KPEX项目编译安装前需要先进行 PyTorch 编译安装，推荐使用环境管理工具，便于对依赖包进行隔离和管理。推荐[下载](https://repo.anaconda.com/miniconda)与[安装使用](https://www.anaconda.com/docs/getting-started/miniconda/install/overview) Miniconda 工具

#### 2. [获取 HPCKit 软件包](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_007.html)
> https://www.hikunpeng.com/developer/hpc/hpckit-download

#### 3. [安装 HPCKit](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_012.html)

##### 解压 HPCKit 软件包（HPCKit 版本号根据实际情况调整）
```
tar xvf HPCKit_26.0.RC1_Linux-aarch64.tar.gz
```
##### 安装 HPCKit
```
sh HPCKit_26.0.RC1_Linux-aarch64/install.sh -y --prefix=[HPCKit安装目录]
```

#### 4. [设置环境变量](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_014.html)

##### 安装 Module 环境变量管理工具（需要已配置 yum 源）
```
yum install environment-modules
```

##### 加载 module
```
module use [HPCKit安装目录]/HPCKit/latest/modulefiles
```

##### 加载编译器环境变量（编译器版本号根据实际情况调整）
```
module load bisheng/compiler5.1.0.2/bishengmodule
```

##### 加载依赖库环境变量
```
module load bisheng/kutacc26.0.RC1/kutacc
module load bisheng/kupl26.0.RC1/release
module load bisheng/hmpi26.0.RC1/release
module load bisheng/kudnn26.0.RC1/kudnn
module load bisheng/kml26.0.RC1/kblas/multi
```

#### 5. 源码编译安装

进行 KPEX 编译安装前，需要先完成 [PyTorch 编译安装](requirements_install.md)
- 如果使用源码压缩包的代码下载方式
##### 上传并解压 KPEX 项目源码包（分支名根据实际情况调整）
```
unzip kunpeng-extension-for-pytorch-main.zip
```

##### 进入KPEX项目源码根目录（分支名根据实际情况调整）
```
cd kunpeng-extension-for-pytorch-main
```

##### 编译安装
执行下述指令
```
CFLAGS="-stdlib=libstdc++ -lstdc++" KPEX_BUILD_TYPE=release HMPI_ROOT=<path_to_HPCKit>/HPCKit/latest/hmpi/bisheng/release/hmpi KUTACC_ROOT=<path_to_HPCKit>/HPCKit/latest/kutacc/bisheng KUPL_ROOT=<path_to_HPCKit>/HPCKit/latest/kupl/bisheng/release KUDNN_ROOT=<path_to_HPCKit>/HPCKit/latest/kudnn/bisheng pip install --root-user-action=ignore --no-build-isolation --editable .
```
>  **说明**：<path_to_HPCkit>：HPCkit软件的安装目录，请替换为实际路径。

#### 6. 运行与验证

##### 编译安装验证
执行**python**命令进入python命令行，在命令行中执行：
```
import kpex
```
无报错回显，则说明编译安装成功。

##### 运行测试用例
需要进入examples目录，运行optimization_simple_model_test.py:
```
cd examples
python optimization_simple_model_test.py
```
无报错回显，则说明测试用例运行成功。

## 📖学习教程

若您已学习**编译安装**，对本项目有一定认知，并希望**深入了解和体验项目**，请访问下述详细教程。

1. [开发指南](https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_123.html)：提供详细接口开发指南，从零学习接口功能与开发。

## 🔍目录结构
项目详细目录介绍如下。
```
├── csrc                           # 项目C++层源码目录
│   ├── kudnn                      # 对接KuDNN算子后端
│   ├── tpp                        # 计算融合算子
│   ├── utils                      # 通用功能工具
│   └── kpex.cpp                   # C++接口对接Python层功能实现
├── kpex                           # 项目Python层接口
│   ├── tpp                        # 计算融合算子
│   ├── __init__.py
│   └── frontend.py                # 对外接口
├── examples                       # 项目示例
│   ├── csrc                       # 项目示例C++层代码目录
│   ├── kupl_example               # 对接KUPL后端示例Python层代码目录
│   ├── __init__.py
│   └── optimization_test.py       # 算子优化测试示例
├── thirdparty                     # PyTorch适配补丁
├── LICENSE
├── README.md
└── setup.py                       # 项目工程编译脚本
```

## 🤝联系我们

本项目功能和文档正在持续更新和完善中，建议您关注最新版本。

- **问题反馈**：通过 [【Issues】](https://gitcode.com/kunpengcompute/kunpeng-extension-for-pytorch/issues)提交问题。
- **社区互动**：通过 [【鲲鹏论坛（HPC专区）】](https://www.hikunpeng.com/forum/forum-0187135482144798003-1.html)参与交流。
- **技术专栏**：通过 [【鲲鹏社区】](https://www.hikunpeng.com/developer/techArticles) 获取技术文章，如系列化教程、优秀实践等。