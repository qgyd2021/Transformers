#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    """
    python3 step_4_test_model.py --trained_model_path /data/tianxing/PycharmProjects/Transformers/trained_models/sft_llama2_stack_exchange
    python3 step_4_test_model.py --trained_model_path /data/tianxing/PycharmProjects/Transformers/trained_models/sft_llama2_stack_exchange
    python3 step_4_test_model.py --trained_model_path qgyd2021/sft_llama2_stack_exchange
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trained_model_path',
        # default='YeungNLP/bloom-1b4-zh',
        default=(project_path / "trained_models/bloom-1b4-sft").as_posix(),
        type=str,
    )
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.trained_model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map="auto",
        device_map={"": 0},
        # offload_folder="./offload",
    ).to(args.device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.trained_model_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == "llama" else True,
        padding_side="left"
    )

    text = input('User: ')
    while True:
        text = text.strip()
        text = 'Question: {}\n\nAnswer: '.format(text)
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = input_ids.to(args.device)
        outputs = model.generate(input_ids, max_new_tokens=200, do_sample=True, top_p=0.85, temperature=0.35,
                                 repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
        rets = tokenizer.batch_decode(outputs)
        output = rets[0]
        output = output.strip()
        output = output.replace(tokenizer.bos_token, "")
        output = output.lstrip()
        output = output.replace(text, "")
        output = output.replace(tokenizer.eos_token, "")

        print("LLM: {}".format(output))
        text = input('User: ')


if __name__ == '__main__':
    main()
