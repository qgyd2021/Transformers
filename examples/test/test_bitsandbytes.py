#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://github.com/TimDettmers/bitsandbytes/blob/main/examples/int8_inference_huggingface.py
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


def get_args():
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
