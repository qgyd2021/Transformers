#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import pandas as pd
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_subset", default="train.jsonl", type=str)
    parser.add_argument("--valid_subset", default="valid.jsonl", type=str)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="YeungNLP/firefly-chatglm2-6b",
        # default=(project_path / "trained_models/firefly_chatglm2_6b_intent").as_posix(),
        type=str
    )
    parser.add_argument("--output_file", default="result.xlsx", type=str)

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
        offload_folder="./offload"
    ).to(args.device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == "llama" else True
    )

    result = list()
    with open(args.valid_subset, "r", encoding="utf-8") as f:
        for row in f:
            row = json.loads(row)
            conversation = row["conversation"]
            for x in conversation:
                print(x)
                human = x["human"]
                assistant = x["assistant"]
                text = "[Round 1]\n\n问：{}\n\n答：".format(human)
                input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                    top_p=args.top_p, temperature=args.temperature, repetition_penalty=args.repetition_penalty,
                    eos_token_id=tokenizer.eos_token_id
                )
                outputs = outputs.tolist()[0][len(input_ids[0]):]
                response = tokenizer.decode(outputs)
                response = response.strip().replace(tokenizer.eos_token, "").strip()
                result.append({
                    "prompt": human,
                    "label": assistant,
                    "predict": response
                })
    result = pd.DataFrame(result)
    result.to_excel(args.output_file, index=False, encoding="utf_8_sig")
    return


if __name__ == '__main__':
    main()
