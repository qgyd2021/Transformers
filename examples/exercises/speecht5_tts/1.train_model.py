#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/learn/audio-course/chapter6/fine-tuning
"""
import argparse
from collections import defaultdict
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from datasets import Audio, load_dataset
import evaluate
import huggingface_hub
import numpy as np
from transformers import TrainingArguments, Trainer
from transformers import DefaultDataCollator

from transformers import SpeechT5Processor

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

    parser.add_argument("--pretrained_model_name_or_path", default="microsoft/speecht5_tts", type=str)

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


def extract_all_chars(batch):
    all_text = " ".join(batch["normalized_text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


replacements = [
    ("à", "a"),
    ("ç", "c"),
    ("è", "e"),
    ("ë", "e"),
    ("í", "i"),
    ("ï", "i"),
    ("ö", "o"),
    ("ü", "u"),
]


def cleanup_text(inputs):
    for src, dst in replacements:
        inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
    return inputs


def main():
    args = get_args()

    dataset = load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        split=args.dataset_split,
        cache_dir=args.dataset_cache_dir,
    )
    len(dataset)
    dataset = dataset.cast_column("audio", feature=Audio(sampling_rate=16000))
    processor = SpeechT5Processor.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer = processor.tokenizer

    vocabs = dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dataset.column_names,
    )

    # dataset_vocab = set(vocabs["vocab"][0])
    # tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}
    dataset = dataset.map(cleanup_text)

    speaker_counts = defaultdict(int)
    for speaker_id in dataset["speaker_id"]:
        speaker_counts[speaker_id] += 1

    def select_speaker(speaker_id):
        return 100 <= speaker_counts[speaker_id] <= 400

    dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])

    return


if __name__ == '__main__':
    main()
