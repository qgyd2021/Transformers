#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/docs/transformers/installation
"""
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from huggingface_hub import hf_hub_download
from transformers import AutoConfig


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="bigscience/T0_3B",
        type=str
    )
    parser.add_argument(
        "--cache_dir",
        default=(project_path / "cache/huggingface/hub").as_posix(),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    hf_hub_download(
        repo_id=args.pretrained_model_name_or_path,
        filename="config.json",
        cache_dir=args.cache_dir,
    )

    config = AutoConfig.from_pretrained(os.path.join(args.cache_dir, "config.json"))
    print(config)
    return


if __name__ == '__main__':
    main()
