#!/usr/bin/env bash

# sh make_thirdparty.sh --stage 0 --stop_stage 1


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
  $verbose && echo "stage 0: download transformers"
  cd "${thirdparty_dir}" || exit 1;

  wget https://github.com/huggingface/transformers/archive/refs/tags/v4.30.2.zip
  unzip v4.30.2.zip
  rm -rf v4.30.2.zip

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: download bitsandbytes"
  cd "${thirdparty_dir}" || exit 1;

  wget https://github.com/TimDettmers/bitsandbytes/archive/refs/tags/0.39.1.zip
  unzip 0.39.1.zip
  rm -rf 0.39.1.zip

fi
