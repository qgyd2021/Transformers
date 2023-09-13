#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from project_settings import project_path
"""
单轮对话，不具有对话历史的记忆功能
"""


def get_args():
    """
    python3 4.test_model.py --pretrained_model_name_or_path /data/tianxing/PycharmProjects/Transformers/trained_models/qwen_7b_chinese_modern_poetry

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        # default="YeungNLP/firefly-chatglm2-6b",
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

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        # offload_folder="./offload",
        empty_init=False
    ).to(args.device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == "llama" else True
    )

    # QWenTokenizer比较特殊, pad_token_id, bos_token_id, eos_token_id 均 为None. eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == "QWenTokenizer":
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id

    text = input("User: ")
    while True:
        text = text.strip()
        # chatglm使用官方的数据组织格式
        if model.config.model_type == "chatglm":
            text = "[Round 1]\n\n问：{}\n\n答：".format(text)
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
        # 为了兼容qwen-7b，因为其对eos_token进行tokenize，无法得到对应的eos_token_id
        else:
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
            bos_token_id = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(args.device)
            eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(args.device)
            input_ids = torch.concat([bos_token_id, input_ids, eos_token_id], dim=1)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                top_p=args.top_p, temperature=args.temperature, repetition_penalty=args.repetition_penalty,
                eos_token_id=tokenizer.eos_token_id
            )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(tokenizer.eos_token, "").strip()
        print("LLM: {}".format(response))
        text = input('User: ')


if __name__ == '__main__':
    main()
