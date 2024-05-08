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
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


PROMPT = """
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
        default=PROMPT,
        type=str,
    )
    parser.add_argument(
        "--response",
        default="不知道",
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


def log_prob_of_response(model: GPT2LMHeadModel,
                         tokenizer: GPT2Tokenizer,
                         prompt: str,
                         response: str,
                         max_input_len: int,
                         ) -> torch.FloatTensor:

    prompt_encoded = tokenizer.__call__(prompt, add_special_tokens=True)
    prompt_input_ids: List[int] = prompt_encoded["input_ids"]

    response_encoded = tokenizer.__call__(response, add_special_tokens=False)
    response_input_ids: List[int] = response_encoded["input_ids"]

    input_ids_ = prompt_input_ids + response_input_ids + [tokenizer.sep_token_id]

    prompt_length = len(prompt_input_ids)
    response_length = len(response_input_ids)
    prompt_response_length = len(input_ids_)
    response_index_list = list(range(prompt_length, prompt_response_length))
    response_token_id_list = [input_ids_[response_index] for response_index in response_index_list]

    input_ids = torch.tensor([input_ids_], dtype=torch.long)
    input_ids = input_ids[:, -max_input_len:]

    # generate
    with torch.no_grad():
        outputs: CausalLMOutputWithCrossAttentions = model.__call__(input_ids)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)[0]

    # print(input_ids_)
    # print(prompt_length)
    # print(prompt_response_length)
    # print(response_index_list)
    # print(response_token_id_list)

    response_log_prob = 0.0
    for idx, token_id in zip(response_index_list, response_token_id_list):
        prob = probs[idx][token_id]
        log_prob = torch.log(prob)
        response_log_prob += log_prob

    # response_prob = torch.exp(response_log_prob)
    return response_log_prob


def main():
    args = get_args()

    # pretrained model
    model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer.eos_token = tokenizer.sep_token
    tokenizer.eos_token_id = tokenizer.sep_token_id

    response_list = [
        "不知道",
        "不知道。",
        "开启",
        "关闭",
    ]

    for response in response_list:
        response_log_prob = log_prob_of_response(model, tokenizer, args.prompt, response, args.max_input_len)
        print(response_log_prob)

    return


if __name__ == '__main__':
    main()
