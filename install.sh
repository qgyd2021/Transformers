#!/usr/bin/env bash
# https://blog.csdn.net/bo_self_effacing/article/details/123628224

# sh install.sh --system_version centos --stage 0 --stop_stage 1

system_version="centos";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

#python_version=3.8.10
#python_bin=python3.8

python_version=3.10.12
python_bin=python3.10

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


if [ $system_version == "centos" ]; then

  #yum install -y git-lfs
  if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    $verbose && echo "stage 0: install python"

    cd "${work_dir}" || exit 1;
    sh ./script/install_python.sh --system_version "centos" --python_version "${python_version}"
  fi

  if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    $verbose && echo "stage 1: make virtualenv"

    cd "${work_dir}" || exit 1;

    # /usr/local/python-3.8.10/bin/python3.8 -m pip install --upgrade pip
    # /usr/local/python-3.10.11/bin/python3.10 -m pip install --upgrade pip --trusted-host pypi.org
    # /usr/local/python-3.10.11/bin/virtualenv Transformers
    /usr/local/python-${python_version}/bin/${python_bin} -m pip install --upgrade pip --trusted-host pypi.org
    mkdir -p /data/local/bin
    cd /data/local/bin || exit 1;
    # source /data/local/bin/Transformers/bin/activate
    /usr/local/python-${python_version}/bin/virtualenv Transformers
  fi

fi
