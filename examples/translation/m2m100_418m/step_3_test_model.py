#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="facebook/m2m100_418M",
        # default=(project_path / "trained_models/firefly_chatglm2_6b_intent").as_posix(),
        type=str
    )
    parser.add_argument("--src_text", default="生活就像一盒巧克力。", type=str)
    parser.add_argument("--src_lang", default="zh", type=str)
    # Life is like a box of chocolate.
    parser.add_argument("--tgt_lang", default="en", type=str)

    parser.add_argument("--temperature", default=0.35, type=float)
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model = M2M100ForConditionalGeneration.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer = M2M100Tokenizer.from_pretrained(args.pretrained_model_name_or_path)

    tokenizer.src_lang = args.src_lang
    encoded_src = tokenizer(args.src_text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_src, forced_bos_token_id=tokenizer.get_lang_id(args.tgt_lang))
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    print(result)
    return


if __name__ == '__main__':
    main()
