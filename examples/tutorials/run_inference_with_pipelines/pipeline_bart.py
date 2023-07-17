#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/docs/transformers/pipeline_tutorial
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
        default="automatic-speech-recognition",
        type=str
    )
    parser.add_argument(
        "--model_name",
        default="facebook/bart-large-mnli",
        type=str
    )
    parser.add_argument(
        "--text",
        default="I have a problem with my iphone that needs to be resolved asap!!",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    classifier = pipeline(model=args.model_name)
    result: dict = classifier(
        inputs=args.text,
        candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
    )
    print(result)
    return


if __name__ == '__main__':
    main()
