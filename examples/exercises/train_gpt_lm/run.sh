#!/usr/bin/env bash

# nohup sh run.sh --stage 0 --stop_stage 0 --system_version centos &
# sh run.sh --stage 0 --stop_stage 1 --system_version windows
# sh run.sh --stage 1 --stop_stage 1
# sh run.sh --stage -1 --stop_stage 1

# params
system_version="windows";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

train_subset=train.txt

pretrained_model_name=gpt2-chinese-cluecorpussmall

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
file_dir="$(pwd)/file_dir"
checkpoint_dir="${file_dir}/checkpoint_dir"
pretrained_models_dir="${work_dir}/../../../pretrained_models";
dataset_cache_dir="${file_dir}/cache_dir"

mkdir -p "${file_dir}"
mkdir -p "${checkpoint_dir}"
mkdir -p "${pretrained_models_dir}"
mkdir -p "${dataset_cache_dir}"


export PYTHONPATH="${work_dir}/../../.."


if [ $system_version == "windows" ]; then
  alias python3='C:/Users/tianx/PycharmProjects/virtualenv/Transformers/Scripts/python.exe'
elif [ $system_version == "centos" ]; then
  source /data/local/bin/Transformers/bin/activate
  alias python3='/data/local/bin/Transformers/bin/python3'
elif [ $system_version == "ubuntu" ]; then
  source /data/local/bin/Transformers/bin/activate
  alias python3='/data/local/bin/Transformers/bin/python3'
fi


declare -A pretrained_model_dict
pretrained_model_dict=(
  ["gpt2-chinese-cluecorpussmall"]="https://huggingface.co/uer/gpt2-chinese-cluecorpussmall"
  ["gpt2"]="https://huggingface.co/gpt2"
  ["japanese-gpt2-medium"]="https://huggingface.co/rinna/japanese-gpt2-medium"

)
pretrained_model_dir="${pretrained_models_dir}/${pretrained_model_name}"


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  $verbose && echo "stage -1: download pretrained model"
  cd "${file_dir}" || exit 1;

  if [ ! -d "${pretrained_model_dir}" ]; then
    cd "${pretrained_models_dir}" || exit 1;

    repository_url="${pretrained_model_dict[${pretrained_model_name}]}"
    git clone "${repository_url}"

    cd "${pretrained_model_dir}" || exit 1;
    rm flax_model.msgpack && rm pytorch_model.bin && rm tf_model.h5
    wget "${repository_url}/resolve/main/pytorch_model.bin"
  fi
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 1: fine tune"
  cd "${work_dir}" || exit 1;

  python3 1.train_model.py \
  --train_subset "${file_dir}/${train_subset}" \
  --pretrained_model_dir "${pretrained_model_dir}" \
  --cache_dir "${dataset_cache_dir}" \
  --output_dir "${checkpoint_dir}" \

fi
