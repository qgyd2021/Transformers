#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/docs/transformers/tasks/language_modeling
"""
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

import huggingface_hub
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

import project_settings as settings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="qgyd2021/distilgpt2-eli5-casual-language-model",
        type=str
    )
    parser.add_argument(
        "--prompt",
        default="Somatic hypermutation allows the immune system to",
        type=str
    )
    parser.add_argument(
        "--hf_token",
        default=settings.environment.get("hf_token", default=None, dtype=str),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    huggingface_hub.login(token=args.hf_token)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )
    inputs = tokenizer.__call__(args.prompt, return_tensors="pt").input_ids

    model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    print(outputs)
    return


if __name__ == '__main__':
    main()
