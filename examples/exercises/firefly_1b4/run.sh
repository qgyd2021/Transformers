#!/usr/bin/env bash

# sh run.sh --stage -1 --stop_stage 2 --system_version centos --pretrained_model_name bloom-1b4-zh --final_model_name bloom-1b4-sft
# sh run.sh --stage -1 --stop_stage 1 --system_version centos --pretrained_model_name bloom-1b4-zh
# sh run.sh --stage 1 --stop_stage 1 --system_version centos --pretrained_model_name bloom-1b4-zh
# sh run.sh --stage 2 --stop_stage 2 --system_version centos --pretrained_model_name bloom-1b4-zh

# sh run.sh --stage 1 --stop_stage 1 --system_version windows --pretrained_model_name bloom-1b4-zh

# params
system_version="windows";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5
pretrained_model_supplier=YeungNLP

#pretrained_model_name=bloom-396m-zh
#pretrained_model_name=bloom-820m-zh
pretrained_model_name=bloom-1b4-zh

final_model_name=final_model_name


patience=0


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

data_dir="/data/tianxing/PycharmProjects/datasets/firefly_train_1_1m"
pretrained_models_dir="${work_dir}/../../../pretrained_models/huggingface/${pretrained_model_supplier}"
final_model_dir="${work_dir}/../../../trained_models/${final_model_name}";

mkdir -p "${file_dir}"
mkdir -p "${cache_dir}"
mkdir -p "${serialization_dir}"
mkdir -p "${data_dir}"
mkdir -p "${pretrained_models_dir}"
mkdir -p "${final_model_dir}"

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


function search_best_ckpt() {
  patience="$1";

  cd "${serialization_dir}" || exit 1
  last_epoch=$(ls . | \
               grep "checkpoint-*" | \
               awk -F'[-]' '{print$2}' | \
               sort -n | \
               awk 'END {print}')

  target_dir=
  if [ -n "${last_epoch}" ]; then
    target_epoch=$((last_epoch - patience))

    for epoch_idx in $(ls . | grep "checkpoint-*" | awk -F'[-]' '{print$2}' | sort -nr):
    do
      if [ "${epoch_idx}" -le "${target_epoch}" ]; then
        target_dir="checkpoint-${epoch_idx}";
        break;
      fi
    done
  fi

  echo "${target_dir}"
}


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  $verbose && echo "stage -1: download data"
  cd "${data_dir}" || exit 1;

  firefly_train_1_1m_size=$(/bin/ls -l firefly-train-1.1M.jsonl | awk '{print $5}')
  if [ ! -e firefly-train-1.1M.jsonl ] || [ "${firefly_train_1_1m_size}" != "1171119212" ]; then
    # rm firefly-train-1.1M.jsonl
    wget -c https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M/resolve/main/firefly-train-1.1M.jsonl
  fi

fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: download pretrained model"
  cd "${work_dir}" || exit 1;
  cd "${pretrained_models_dir}" || exit 1;

  if [ ! -d "${pretrained_model_name}" ]; then
    git clone "https://huggingface.co/${pretrained_model_supplier}/${pretrained_model_name}/"

    cd "${pretrained_models_dir}/${pretrained_model_name}" || exit 1;
    rm -rf .git
    rm -rf flax_model.msgpack
    rm -rf model.safetensors
    rm -rf pytorch_model.bin
    rm -rf tokenizer.json

  fi

  cd "${pretrained_models_dir}/${pretrained_model_name}" || exit 1;
  if [ ! -e pytorch_model.bin ]; then
    wget -c "https://huggingface.co/${pretrained_model_supplier}/${pretrained_model_name}/resolve/main/pytorch_model.bin"
  fi

  if [ ! -e tokenizer.json ]; then
    wget -c "https://huggingface.co/${pretrained_model_supplier}/${pretrained_model_name}/resolve/main/tokenizer.json"
  fi

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: train model"
  cd "${work_dir}" || exit 1;
  target_dir=$(search_best_ckpt "${patience}");

  resume_from_checkpoint=
  if [ -n "${target_dir}" ]; then
  resume_from_checkpoint="${serialization_dir}/${target_dir}"
    echo "resume_from_checkpoint: ${resume_from_checkpoint}"
  fi

  python3 1.train_model.py \
  --train_file "${data_dir}/firefly-train-1.1M.jsonl" \
  --pretrained_model_name_or_path "${pretrained_models_dir}/${pretrained_model_name}" \
  --output_dir "${serialization_dir}" \
  --cache_dir "${cache_dir}" \
  --fp16 \
  ${resume_from_checkpoint:+--resume_from_checkpoint $resume_from_checkpoint}

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: collect files"
  target_dir=$(search_best_ckpt "${patience}");

  cd "${work_dir}" || exit 1;

  cp "${serialization_dir}/${target_dir}/pytorch_model.bin" "${final_model_dir}/pytorch_model.bin"

  cp "${pretrained_models_dir}/${pretrained_model_name}/config.json" "${final_model_dir}/config.json"
  cp "${pretrained_models_dir}/${pretrained_model_name}/special_tokens_map.json" "${final_model_dir}/special_tokens_map.json"
  cp "${pretrained_models_dir}/${pretrained_model_name}/tokenizer_config.json" "${final_model_dir}/tokenizer_config.json"
  cp "${pretrained_models_dir}/${pretrained_model_name}/tokenizer.json" "${final_model_dir}/tokenizer.json"

fi
