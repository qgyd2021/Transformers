#!/usr/bin/env bash

#bitsandbytes
#https://github.com/TimDettmers/bitsandbytes
#
### 安装bitsandbytes
#
#bitsandbytes 是 CUDA 自定义函数的轻量级包装器, 特别是 8 位优化器, 矩阵乘法 (LLM.int8()) 和量化函数.
#
#### 安装
#
#通过 `pip3 install bitsandbytes` 来安装.
#
#安装之后通过 `python -m bitsandbytes` 来验证安装是否成功.
#
#在某些情况下可能需要从源代码进行编译.
#
#```text
#git clone https://github.com/timdettmers/bitsandbytes.git
#cd bitsandbytes
#
## CUDA_VERSIONS in {110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 120}
## make argument in {cuda110, cuda11x, cuda12x}
## if you do not know what CUDA you have, try looking at the output of: python -m bitsandbytes
#CUDA_VERSION=117 make cuda11x
#python setup.py install
#```
#
#### 备注
#
##### 必须安装与 GPU 版本相匹配的 CUDA
#
#我的情况如下:
#
#**GPU 和 CUDA 版本. **
#
#```text
## nvidia-smi
#Mon Aug 28 14:38:32 2023
#+-----------------------------------------------------------------------------+
#| NVIDIA-SMI 515.105.01   Driver Version: 515.105.01   CUDA Version: 11.7     |
#|-------------------------------+----------------------+----------------------+
#| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
#| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
#|                               |                      |               MIG M. |
#|===============================+======================+======================|
#|   0  Tesla V100S-PCI...  Off  | 00000000:0B:00.0 Off |                    0 |
#| N/A   48C    P0    40W / 250W |  14154MiB / 32768MiB |      0%      Default |
#|                               |                      |                  N/A |
#+-------------------------------+----------------------+----------------------+
#
#+-----------------------------------------------------------------------------+
#| Processes:                                                                  |
#|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
#|        ID   ID                                                   Usage      |
#|=============================================================================|
#|    0   N/A  N/A     11127      C   python3                         12973MiB |
#|    0   N/A  N/A     25921      C   python3                          1177MiB |
#+-----------------------------------------------------------------------------+
#```
#
#**CUDA 版本. **
#
#我的经历, 安装 nvidia driver 驱动后 PyTorch 就可以使用 GPU 了, 同时 `nvidia-smi` 命令中也会显示 `CUDA Version: 11.7`.
#
#但是 `/usr/local/cuda:/usr/local/cuda-11.7` 是不存在的. 这个需要单独安装 (即: 安装 CUDA).
#
#```text
## ll /usr/local/ | grep cuda
#lrwxrwxrwx   1 root root  20 Aug 15 19:12 cuda -> /usr/local/cuda-11.7
#drwxr-xr-x  14 root root 268 Aug 15 18:31 cuda-11.7
#```
#
##### 从编译安装
#
#从编译安装使用的命令如下:
#
#```
#CUDA SETUP: Something unexpected happened. Please compile from source:
#git clone git@github.com:TimDettmers/bitsandbytes.git
#cd bitsandbytes
#CUDA_VERSION=117 make cuda11x_nomatmul
#python setup.py install
#```
#
#我的情况是没有使用容器, 在宿主机上安装的.
#
#1. 之前机器上安装的是 `/usr/local/cuda-10.4` 编译不通过. 因为 `/usr/local/cuda-10.4/bin` 下的 `nvcc` 编译器与 GPU 所需的 `CUDA Version: 11.7` 是不匹配的.
#2. 后来安装了 `/usr/local/cuda-11.7` 并删除 `/usr/local/cuda-10.4`, 但还是安装不成功.
#
#偶然的一次尝试:
#
#**需要使用conda虚拟环境**, 在 python 的 virtualenv 中安装失败了.
#
#**执行 `CUDA_VERSION=117 make cuda11x_nomatmul` 命令时, 确保以下几项正确**. 即:
#
#* `NVCC path`: 指向了 cuda 中的 nvcc 编译器. (`nvcc` 是 cuda 提供的一款编译器).
#
#* `CUDA_HOME`: cuda 安装的目录, 一般安装时会自动确定在 `/usr/local/cuda`,
#
#* `CONDA_PREFIX`: 是 `conda` 下创建的虚拟环境.
#
#* `PATH`: 应包含 cuda 的 bin 目录.
#
#* `LD_LIBARY_PATH`: 应包含 cuda 的 lib 目录.
#
#```
#(Transformers) [root@nlp bitsandbytes-0.39.1]# CUDA_VERSION=117 make cuda11x_nomatmul
#ENVIRONMENT
#============================
#CUDA_VERSION: 117
#============================
#NVCC path: /usr/local/cuda/bin/nvcc
#GPP path: /usr/bin/g++ VERSION: g++ (GCC) 11.1.0
#CUDA_HOME: /usr/local/cuda
#CONDA_PREFIX: /usr/local/miniconda3/envs/Transformers
#PATH: /usr/local/miniconda3/envs/Transformers/bin:/usr/local/miniconda3/condabin:/usr/local/sbin:/sbin:/bin:/usr/sbin:/usr/bin:/root/bin:/usr/local/cuda/bin:/root/bin
#LD_LIBRARY_PATH: /usr/local/cuda/lib64
#============================
#```
#
#我的情况是, 本应该是 `LD_LIBRARY_PATH: /usr/local/cuda/lib64` 的项变成了 `LD_LIBRARY_PATH:`.
#
#检查 `cat ~/.bashrc` 中包含:
#
#```text
#CUDA_HOME="/usr/local/cuda"
#PATH=/usr/local/sbin:/sbin:/bin:/usr/sbin:/usr/bin:/root/bin:/usr/local/miniconda3/bin:/usr/local/cuda/bin
#LD_LIBRARY_PATH=/usr/local/cuda/lib64
#```
#
#同时再执行
#
#```text
#export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
#```
#
#之后就编译成功了.
#
#**检查是否安装成功**
#
#重新连接 Terminal 之后, 在执行 `python -m bitsandbytes` 之前先执行以下命令.
#
#```text
#export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
#```
#
#这一步非常奇怪, 因为 `echo $LD_LIBRARY_PATH`, 可以看到 `/usr/local/cuda/lib64` 路径在其中.  `echo $PATH` 都可以看到 `/usr/local/cuda` 在其中.
#
#但是执行 `CUDA_VERSION=117 make cuda11x_nomatmul` 时会发现原本应该是 `LD_LIBRARY_PATH: /usr/local/cuda/lib64` 的项变成了 `LD_LIBRARY_PATH:`.
#
#只要再执行一次以上命名, 再执行以下命令, 就可以成功.
#
#安装后执行以下命令, 检查是否安装成功.
#
#```
#python -m bitsandbytes
#```
#
#如过出现以下内容, 说明安装成功了.
#
#```text
#...
#...
#...
#Running a quick check that:
#    + library is importable
#    + CUDA function is callable
#
#
#WARNING: Please be sure to sanitize sensible info from any such env vars!
#
#SUCCESS!
#Installation was successful!
#```


# sh install_bitsandbytes.sh --stage 0 --stop_stage 0


verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5


# parse options
while true; do
  [ -z "${1:-}" ] && break;  # break if there are no arguments
  case "$1" in
    --*) name=$(echo "$1" | sed s/^--// | sed s/-/_/g);
      eval '[ -z "${'"$name"'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1;
      old_value="(eval echo \\$$name)";
      if [ "${old_value}" == "true" ] || [ "${old_value}" == "false" ]; then
        was_bool=true;
      else
        was_bool=false;
      fi

      # Set the variable to the right value-- the escaped quotes make it work if
      # the option had spaces, like --cmd "queue.pl -sync y"
      eval "${name}=\"$2\"";

      # Check that Boolean-valued arguments are really Boolean.
      if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\": $1 $2" 1>&2
        exit 1;
      fi
      shift 2;
      ;;

    *) break;
  esac
done


work_dir="$(pwd)"
thirdparty_dir="${work_dir}/thirdparty"

mkdir -p "${thirdparty_dir}"


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: download bitsandbytes"
  cd "${thirdparty_dir}" || exit 1;

  wget https://github.com/TimDettmers/bitsandbytes/archive/refs/tags/0.39.1.zip
  unzip 0.39.1.zip
  rm -rf 0.39.1.zip

fi


