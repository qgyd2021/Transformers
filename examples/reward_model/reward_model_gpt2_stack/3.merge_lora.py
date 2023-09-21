#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
"""
使用该脚本，将lora的权重合并大base model中
"""


def get_args():
    """
    python3 3.merge_lora.py \
    --pretrained_model_name_or_path /data/tianxing/PycharmProjects/Transformers/pretrained_models/huggingface/gpt2 \
    --adapter_name_or_path /data/tianxing/PycharmProjects/Transformers/examples/reward_model/reward_model_gpt2_stack/gpt2_peft_stack-exchange-paired_rmts__100000_2e-05_peft_last_checkpoint \
    --save_directory /data/tianxing/PycharmProjects/Transformers/trained_models/reward_model_gpt2_stack

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="YeungNLP/firefly-chatglm2-6b",
        type=str
    )
    parser.add_argument(
        "--adapter_name_or_path",
        default="YeungNLP/firefly-baichuan-7b-qlora-sft",
        type=str
    )
    parser.add_argument("--save_directory", default="save_directory", type=str)

    parser.add_argument("--num_labels", default="num_labels", type=int)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    config = AutoConfig.from_pretrained(
        args.pretrained_model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model_name_or_path,
        num_labels=args.num_labels,
    )

    model = PeftModel.from_pretrained(model, args.adapter_name_or_path, device_map={"": "cpu"})
    model = model.merge_and_unload()

    tokenizer.save_pretrained(args.save_directory)
    model.save_pretrained(args.save_directory)
    return


if __name__ == '__main__':
    main()
