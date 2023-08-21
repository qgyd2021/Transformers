#!/usr/bin/env bash


# sh run.sh --stage -1 --stop_stage -1 --system_version windows


# params
system_version="windows";
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


$verbose && echo "system_version: ${system_version}"

work_dir="$(pwd)"
file_dir="${work_dir}/file_dir"
data_dir="${work_dir}/data_dir"
cache_dir="${file_dir}/cache_dir"
serialization_dir="${file_dir}/serialization_dir"

mkdir -p "${file_dir}"
mkdir -p "${cache_dir}"
mkdir -p "${serialization_dir}"
mkdir -p "${data_dir}"

export PYTHONPATH="${work_dir}/../../.."

if [ $system_version == "windows" ]; then
  alias python3='C:/Users/tianx/PycharmProjects/virtualenv/Transformers/Scripts/python.exe'
elif [ $system_version == "centos" ]; then
  # conda activate Transformers
  alias python3='/usr/local/miniconda3/envs/Transformers/bin/python3'
elif [ $system_version == "ubuntu" ]; then
  # conda activate Transformers
  alias python3='/usr/local/miniconda3/envs/Transformers/bin/python3'
fi



if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  $verbose && echo "stage -1: download data"
  cd "${data_dir}" || exit 1;

  data_name_array=(
    dev/dev_part_0.tar.gz
    test/test_part_0.tar.gz
    train/train_part_0.tar.gz
    train/train_part_1.tar.gz
    train/train_part_2.tar.gz
    train/train_part_3.tar.gz
    train/train_part_4.tar.gz
    asr_dev.tsv
    asr_test.tsv
    asr_train.tsv
  )

  for data_name in ${data_name_array[*]}
  do
    url="https://huggingface.co/datasets/facebook/voxpopuli/resolve/main/data/nl/${data_name}"

    # https://www.jb51.net/article/275532.htm
    path=${data_name%/*}
    name=${data_name##*/}
    un_tar_dir=${data_name%%.*}

    if [ ! -e "${data_name}" ]; then
      echo "${data_name}"
      if [ "${path}" == "${name}" ]; then
        wget -c "${url}"
      else
        mkdir -p "${path}"
        wget -c -P "${path}" "${url}"
      fi
    fi

    if [ ! -d "${un_tar_dir}" ]; then

      if [ "${path}" != "${name}" ]; then
        tar -zxvf "${data_name}" -C "${path}"
      fi
    fi

  done
fi
