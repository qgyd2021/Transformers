#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import platform
import re
import string
from typing import List

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

import torch
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


def get_args():
    """
    python3 step_3_test_model.py --pretrained_model_name_or_path /data/tianxing/PycharmProjects/Transformers/pretrained_models/huggingface/qgyd2021/lip_service_4chan
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="qgyd2021/lip_service_4chan",
        type=str,
    )
    parser.add_argument("--max_input_len", default=512, type=int)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--temperature", default=0.35, type=float)
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str)

    args = parser.parse_args()
    return args


def remove_space_between_cn_en(text):
    splits = re.split(" ", text)
    if len(splits) < 2:
        return text

    result = ""
    for t in splits:
        if t == "":
            continue
        if re.search(f"[a-zA-Z0-9{string.punctuation}]$", result) and re.search("^[a-zA-Z0-9]", t):
            result += " "
            result += t
        else:
            if not result == "":
                result += t
            else:
                result = t

    if text.endswith(" "):
        result += " "
    return result


def main():
    args = get_args()

    # pretrained model
    model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer.eos_token = tokenizer.sep_token
    tokenizer.eos_token_id = tokenizer.sep_token_id

    # chat
    while True:
        text = input("text:")
        text: str = str(text).strip()
        if text == "Quit":
            break

        text_encoded = tokenizer.__call__(text, add_special_tokens=True)
        input_ids: List[int] = text_encoded["input_ids"]
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        input_ids = input_ids[:, -args.max_input_len:]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=tokenizer.sep_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        answer = tokenizer.decode(outputs)
        answer = answer.strip().replace(tokenizer.eos_token, "").strip()
        answer = answer.strip().replace(tokenizer.unk_token, "").strip()
        answer = answer.strip().replace(tokenizer.cls_token, "").strip()
        answer = answer.strip().replace(tokenizer.sep_token, "").strip()

        answer = remove_space_between_cn_en(answer)
        print(answer)

    return


if __name__ == '__main__':
    main()
