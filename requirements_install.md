# PyTorch 编译指南

## 🚀概述

本文描述适配 Kunpeng Extension for PyTorch（以下简称KPEX）优化的 PyTorch 编译安装流程。

## 📝版本配套

- 运行平台
    - 鲲鹏 920 专业版
- 系统规格
    - openEuler 22.03（LTS-SP4）AArch64
- 软件依赖
    - Python >= 3.11.0

## ⚡️编译安装

#### 1. 环境准备

在进行 PyTorch 编译安装之前，需要先进行 [HPCKit 软件包安装与环境变量设置](README.md/#2-获取-hpckit-软件包)

#### 2. 获取源码

PyTorch源码及所有子模块获取命令如下：
```
git clone -b v2.5.0 --depth=1 --recursive <https://github.com/pytorch/pytorch.git>
cd pytorch  
git submodule update --init --recursive
```
或者在官网Release链接中[下载PyTorch源码包](https://github.com/pytorch/pytorch/releases/download/v2.5.0/pytorch-v2.5.0.tar.gz)，下载完成后上传软件包并进行解压


#### 3. 补丁安装

##### 获取补丁
安装适配Kunpeng HPCKit优化的PyTorch需要安装KML适配与优化补丁和libnop编译问题修改补丁，补丁获取路径请参见：
[KML适配与优化补丁](thirdparty/pytorch-v2.5.0.patch)
[libnop编译问题修改补丁](thirdparty/pytorch-v2.5.0-thirdparty-libnop.patch)

##### 安装补丁
下载上述补丁后，将KML适配与优化补丁复制到PyTorch源码根目录<path_to_pytorch-v2.5.0>下并在该目录运行：
```
git apply pytorch-v2.5.0.patch
```

将libnop编译问题修改补丁复制到PyTorch第三方软件libnop的目录<path_to_pytorch-v2.5.0>/third_party/tensorpipe/third_party/libnop下并在该目录运行：
```
git apply pytorch-v2.5.0-thirdparty-libnop.patch
```

#### 4. PyTorch 编译安装


##### 进入PyTorch源码根目录
```
cd <path_to_pytorch-v2.5.0>
```

##### 安装PyTorch所需依赖
```
pip install -r requirements.txt
```

##### 运行编译指令
```
CFLAGS="-L <path_to_HPCKit>/HPCKit/latest/hmpi/bisheng/release/hmpi/lib -lmpi -stdlib=libstdc++ -lstdc++" MAX_JOBS=128 SVE_VECBITS=512 DEBUG=0 USE_MKLDNN=0 USE_CUDA=0 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=1 USE_QNNPACK=0 USE_XNNPACK=0 USE_HBM=0 BLAS=KML USE_LAPACK=1 KML_LAPACK_LIB=<path_to_HPCKit>/HPCKit/latest/kml/bisheng/lib/sve512 KML_LAPACK_INCLUDE=<path_to_HPCKit>/HPCKit/latest/kml/bisheng/include python ./setup.py develop
```
>  **说明**：<path_to_HPCkit>：HPCkit软件的安装目录，请替换为实际路径。

#### 5. 运行与验证

执行**python**命令进入python命令行，在命令行中执行：
```shell
import torch
```
无报错回显，则说明编译安装成功。