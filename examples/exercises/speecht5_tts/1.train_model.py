#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/learn/audio-course/chapter6/fine-tuning
"""
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from datasets import Audio, load_dataset
import evaluate
import huggingface_hub
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer
from transformers import DefaultDataCollator

import project_settings as settings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="facebook/voxpopuli", type=str)
    parser.add_argument("--dataset_name", default="nl", type=str)
    parser.add_argument("--dataset_split", default="train", type=str)
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )

    parser.add_argument(
        "--file_dir",
        default="file_dir",
        type=str
    )

    parser.add_argument(
        "--hub_strategy",
        default="end",
        choices=["end", "every_save", "checkpoint", "all_checkpoints"],
        type=str
    )
    parser.add_argument(
        "--hf_token",
        default=settings.environment.get("hf_token", default=None, dtype=str),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    dataset = load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        split=args.dataset_split,
        cache_dir=args.dataset_cache_dir,
    )
    len(dataset)
    # dataset = dataset.cast_column("audio", feature=Audio(sampling_rate=16000))

    return


if __name__ == '__main__':
    main()
