#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
"""
使用该脚本，将lora的权重合并大base model中
"""


def get_args():
    """
    python3 step_3_merge_lora.py \
    --pretrained_model_name_or_path /data/tianxing/PycharmProjects/Transformers/pretrained_models/huggingface/Qwen/Qwen1.5-1.8B \
    --adapter_name_or_path /data/tianxing/PycharmProjects/Transformers/examples/2b_llm/qwen1.5_1.8b_few_shot_intent/file_dir/serialization_dir/checkpoint-137000 \
    --save_directory /data/tianxing/PycharmProjects/Transformers/trained_models/qwen1.5_1.8B_few_shot_intent

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

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map='auto',
        device_map={"": "cpu"}
    )
    model = PeftModel.from_pretrained(model, args.adapter_name_or_path, device_map={"": "cpu"})
    model = model.merge_and_unload()

    tokenizer.save_pretrained(args.save_directory)
    model.save_pretrained(args.save_directory, max_shard_size="2GB")
    return


if __name__ == '__main__':
    main()
