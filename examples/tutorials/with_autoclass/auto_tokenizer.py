#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="bert-base-uncased",
        type=str
    )
    parser.add_argument(
        "--sequence",
        default="In a hole in the ground there lived a hobbit.",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )

    print(tokenizer(args.sequence))
    return


if __name__ == '__main__':
    main()
