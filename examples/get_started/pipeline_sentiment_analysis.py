#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/docs/transformers/quicktour
"""
import argparse
import os
from typing import List

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from transformers import pipeline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="sentiment-analysis",
        type=str
    )
    parser.add_argument(
        "--text1",
        default="We are very happy to show you the 🤗 Transformers library.",
        type=str
    )
    parser.add_argument(
        "--text2",
        default="We hope you don't hate it.",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    classifier = pipeline(task=args.task)

    results: List[dict] = classifier(args.text1)
    for result in results:
        print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

    results: List[dict] = classifier([args.text1, args.text2])
    for result in results:
        print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

    return


if __name__ == '__main__':
    main()
