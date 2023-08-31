#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

from project_settings import project_path


def get_args():
    """
    python3 3.test_model.py --pretrained_model_name_or_path /data/tianxing/PycharmProjects/Transformers/pretrained_models/Baichuan-13B-Chat

    python3 3.test_model.py --pretrained_model_name_or_path /data/tianxing/PycharmProjects/Transformers/trained_models/firefly_chatglm2_6b_intent

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        # default="baichuan-inc/Baichuan-13B-Chat",
        default=(project_path / "trained_models/firefly_chatglm2_6b_intent").as_posix(),
        type=str
    )
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--temperature", default=0.35, type=float)
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        use_fast=False,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="./offload"
    ).to(args.device).eval()
    generation_config = GenerationConfig.from_pretrained(args.pretrained_model_name_or_path)
    model.generation_config = generation_config

    while True:
        # 世界上第二高的山峰是哪座
        text = input("User: ")
        text = text.strip()

        if text == "Quit":
            break

        messages = [{"role": "user", "content": text}]
        response = model.chat(tokenizer, messages)
        print("LLM: {}".format(response))
    return


if __name__ == '__main__':
    main()
