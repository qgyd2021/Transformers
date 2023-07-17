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

from datasets import load_dataset, Audio
from transformers import pipeline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="automatic-speech-recognition",
        type=str
    )
    parser.add_argument(
        "--model",
        default="facebook/wav2vec2-base-960h",
        type=str
    )
    parser.add_argument("--dataset_path", default="PolyAI/minds14", type=str)
    parser.add_argument("--dataset_name", default="en-US", type=str)
    parser.add_argument("--dataset_split", default="train", type=str)
    parser.add_argument("--dataset_column", default="audio", type=str)
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    speech_recognizer = pipeline(
        task=args.task,
        model=args.model,
    )

    dataset = load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        split=args.dataset_split,
        cache_dir=args.dataset_cache_dir
    )
    dataset = dataset.cast_column(
        column=args.dataset_column,
        feature=Audio(
            sampling_rate=speech_recognizer.feature_extractor.sampling_rate
        )
    )
    results = speech_recognizer(dataset[:4]["audio"])
    for result in results:
        print(result["text"])

    return


if __name__ == '__main__':
    main()
