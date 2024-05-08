#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import platform
import re
import string
from typing import List

if platform.system() == "Windows":
    from project_settings import project_path
else:
    project_path = os.path.abspath("./")
    project_path = Path(project_path)

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

import torch
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


prompt = """
意图识别。不确定时输出：不知道。

Examples:
------------
text: 打开风扇
intent: 开启
------------
text: 关闭电视
intent: 关闭
------------
text: 把风扇关了吧
intent: 关闭
------------
text: 电视开开
intent: 开启
------------
text: 天上一天
intent:
"""


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="qgyd2021/few_shot_intent",
        type=str,
    )
    parser.add_argument(
        "--prompt",
        default=prompt,
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

    prompt_encoded = tokenizer.__call__(args.prompt, add_special_tokens=True)
    input_ids: List[int] = prompt_encoded["input_ids"]
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    input_ids = input_ids[:, -args.max_input_len:]

    tokenizer.eos_token = tokenizer.sep_token
    tokenizer.eos_token_id = tokenizer.sep_token_id

    # generate
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
        answer = remove_space_between_cn_en(answer)
        print(answer)

    return


if __name__ == '__main__':
    main()
