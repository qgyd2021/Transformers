#!/usr/bin/env bash

# sh run.sh --stage 0 --stop_stage 0 --system_version centos
# sh run.sh --stage 1 --stop_stage 1 --system_version centos
# sh run.sh --stage 2 --stop_stage 2 --system_version centos

# bitsandbytes
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# params
system_version="windows";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

pretrained_model_supplier=
pretrained_model_name=gpt2

last_checkpoint_dir=last_checkpoint
last_model_name=reward_model_gpt2_stack

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
cache_dir="${file_dir}/cache_dir"
serialization_dir="${file_dir}/serialization_dir"

pretrained_models_dir="${work_dir}/../../../pretrained_models/huggingface/${pretrained_model_supplier}"
last_checkpoint_dir="${work_dir}/../../../trained_models/${last_model_name}";

mkdir -p "${file_dir}"
mkdir -p "${cache_dir}"
mkdir -p "${serialization_dir}"
mkdir -p "${pretrained_models_dir}"
mkdir -p "${last_checkpoint_dir}"

export PYTHONPATH="${work_dir}/../../.."

if [ $system_version == "windows" ]; then
  alias python3='C:/Users/tianx/PycharmProjects/virtualenv/Transformers/Scripts/python.exe'
elif [ $system_version == "centos" ]; then
  # conda activate Transformers
  alias python3='/usr/local/miniconda3/envs/Transformers/bin/python3'
elif [ $system_version == "ubuntu" ]; then
  alias python3='/usr/local/miniconda3/envs/Transformers/bin/python3'
elif [ $system_version == "macos" ]; then
  alias python3='/Users/honey/PycharmProjects/virtualenv/TrainLLM/bin/python'
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: download pretrained model"
  cd "${pretrained_models_dir}" || exit 1;

  if [ ! -d "${pretrained_model_name}" ]; then
    git clone "https://huggingface.co/${pretrained_model_supplier:+$pretrained_model_supplier/}${pretrained_model_name}/"

    cd "${pretrained_model_name}" || exit 1;

    rm -rf onnx/
    rm -rf .git
    rm -rf .gitattributes
    rm -rf 64-8bits.tflite
    rm -rf 64-fp16.tflite
    rm -rf 64.tflite
    rm -rf flax_model.msgpack
    rm -rf model.safetensors
    rm -rf rust_model.ot
    rm -rf tf_model.h5
    rm -rf model.safetensors

  fi

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: prepare data"
  cd "${work_dir}" || exit 1;

  python3 1.prepare_data.py

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: train model"
  cd "${work_dir}" || exit 1;

  python3 2.train_model.py \
  --model_name "${pretrained_models_dir}/${pretrained_model_name}" \
  --last_checkpoint "${last_checkpoint_dir}" \
  --output_dir "${serialization_dir}"

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  $verbose && echo "stage 3: merge lora"
  cd "${work_dir}" || exit 1;

  python3 3.merge_lora.py \
  --pretrained_model_name_or_path "${pretrained_models_dir}/${pretrained_model_name}" \
  --adapter_name_or_path "${serialization_dir}/${last_checkpoint_dir}" \
  --save_directory "${final_model_dir}"

fi
