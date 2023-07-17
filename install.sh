#!/usr/bin/env bash


system_version="centos";
work_dir="$(pwd)"


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


# 下载 transformers 代码, 可以查看官方的 examples 文件.
mkdir -p thirdparty
cd "${work_dir}/thirdparty" || exit 1;
wget https://github.com/huggingface/transformers/archive/refs/tags/v4.27.1.zip
unzip v4.27.1.zip && rm v4.27.1.zip;


if [ $system_version == "centos" ]; then
  python_version=3.8.10

  yum install -y git-lfs

  cd "${work_dir}" || exit 1;
  sh ./script/install_python.sh --system_version "centos" --python_version "${python_version}"

  /usr/local/python-${python_version}/bin/pip3 install virtualenv==23.0.1
  mkdir -p /data/local/bin
  cd /data/local/bin || exit 1;
  # source /data/local/bin/Transformers/bin/activate
  /usr/local/python-${python_version}/bin/virtualenv Transformers
fi
