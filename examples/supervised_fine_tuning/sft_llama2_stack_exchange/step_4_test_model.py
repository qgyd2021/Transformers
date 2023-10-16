#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from project_settings import project_path


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trained_model_path',
        # default='YeungNLP/bloom-1b4-zh',
        default=(project_path / "trained_models/bloom-1b4-sft").as_posix(),
        type=str,
    )
    parser.add_argument('--device', default='auto', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map="auto",
        device_map={"": 0},
        # offload_folder="./offload",
    ).to(args.device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == "llama" else True,
        padding_side="left"

    )

    text = input('User: ')
    while True:
        text = 'Question: {}\n\nAnswer: '.format(text)
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = input_ids.to(args.device)
        outputs = model.generate(input_ids, max_new_tokens=200, do_sample=True, top_p=0.85, temperature=0.35,
                                 repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
        rets = tokenizer.batch_decode(outputs)
        output = rets[0]
        print("LLM: {}".format(output))
        text = input('User: ')


if __name__ == '__main__':
    main()
