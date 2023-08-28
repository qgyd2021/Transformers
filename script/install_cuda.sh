#!/usr/bin/env bash

# 查看系统架构 Architecture
# >>> uname -a
# Linux nlp 3.10.0-1160.66.1.el7.x86_64 #1 SMP Wed May 18 16:02:34 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
# >>> uname -m
# x86_64


#cuda驱动就像普通的软件一样, 可以安装多个.


#在以下路径找到对应版本, 获得安装命令.
#https://developer.nvidia.com/cuda-toolkit-archive
#
#参考链接:
#https://www.cnblogs.com/yuezc/p/12937239.html
#https://blog.csdn.net/pursuit_zhangyu/article/details/117073126
#
#[root@nlp dep]# sh cuda_10.2.89_440.33.01_linux.run --override
#(执行以上命令后, 安提示操作, 以下是安装完成后的信息).
#===========
#= Summary =
#===========
#
#Driver:   Installed
#Toolkit:  Installed in /usr/local/cuda-10.2/
#Samples:  Installed in /home/admin/, but missing recommended libraries
#
#Please make sure that
# -   PATH includes /usr/local/cuda-10.2/bin
# -   LD_LIBRARY_PATH includes /usr/local/cuda-10.2/lib64, or, add /usr/local/cuda-10.2/lib64 to /etc/ld.so.conf and run ldconfig as root
#
#To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-10.2/bin
#To uninstall the NVIDIA Driver, run nvidia-uninstall
#
#Please see CUDA_Installation_Guide_Linux.pdf in /usr/local/cuda-10.2/doc/pdf for detailed information on setting up CUDA.
#Logfile is /var/log/cuda-installer.log


# params:
system_version="centos";


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


echo "system_version: ${system_version}";


if [ ${system_version} = "centos" ]; then
  #runfile(local)
  wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
  sudo sh cuda_11.7.0_515.43.04_linux.run --override

  #只选择安装 CUDA Toolkit 11.7 其它取消选择.

  rm -rf /usr/local/cuda
  ln -snf /usr/local/cuda-11.7 /usr/local/cuda

  #export CUDA_HOME=/usr/local/cuda
  #export PATH="${CUDA_HOME}/bin${PATH:+:$PATH}"
  #export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  #export PATH=$PATH:/usr/local/cuda/bin
  #export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

  cat ~/.bashrc
  echo "PATH=$PATH:/usr/local/cuda/bin" >> /root/.bashrc
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> /root/.bashrc
  source ~/.bashrc

  #查看cuda版本
  nvcc -V

fi
