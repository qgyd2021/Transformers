#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://github.com/TimDettmers/bitsandbytes/blob/main/examples/int8_inference_huggingface.py

bitsandbytes 安装之后, 可以通过 python -m bitsandbytes 命令验证安装是否成功. 并查看提示信息.


失败时, 可能需要自行编译 (conda 环境):
CUDA SETUP: Something unexpected happened. Please compile from source:
git clone git@github.com:TimDettmers/bitsandbytes.git
cd bitsandbytes
CUDA_VERSION=117 make cuda11x_nomatmul
python setup.py install

执行 `CUDA_VERSION=117 make cuda11x_nomatmul` 时, 确保以下几项正确.
(Transformers) [root@nlp bitsandbytes-0.39.1]# CUDA_VERSION=117 make cuda11x_nomatmul
ENVIRONMENT
============================
CUDA_VERSION: 117
============================
NVCC path: /usr/local/cuda/bin/nvcc
GPP path: /usr/bin/g++ VERSION: g++ (GCC) 11.1.0
CUDA_HOME: /usr/local/cuda
CONDA_PREFIX: /usr/local/miniconda3/envs/Transformers
PATH: /usr/local/miniconda3/envs/Transformers/bin:/usr/local/miniconda3/condabin:/usr/local/sbin:/sbin:/bin:/usr/sbin:/usr/bin:/root/bin:/usr/local/cuda/bin:/root/bin
LD_LIBRARY_PATH: /usr/local/cuda/lib64:/usr/local/cuda/lib64
============================





export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

"""
import argparse
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import pipeline, LlamaTokenizer, LlamaForCausalLM


def get_args():
    """
    python3 test_bitsandbytes.py --pretrained_model_name_or_path /data/tianxing/PycharmProjects/Transformers/trained_models/bloom-1b4-sft
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", default="decapoda-research/llama-7b-hf", type=str)
    parser.add_argument("--text", default="Hamburg is in which country?\n", type=str)
    parser.add_argument("--max_new_tokens", default=128, type=int)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    n_gpus = torch.cuda.device_count()

    free_in_gb = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
    max_memory = {i: max_memory for i in range(n_gpus)}

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        device_map='auto',
        load_in_8bit=True,
        max_memory=max_memory
    )

    input_ids = tokenizer(args.text, return_tensors="pt").input_ids

    generated_ids = model.generate(input_ids, max_length=args.max_new_tokens)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    return


if __name__ == '__main__':
    main()
