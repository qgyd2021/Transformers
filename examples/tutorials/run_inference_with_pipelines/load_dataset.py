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

from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="automatic-speech-recognition",
        type=str
    )
    parser.add_argument(
        "--model_name",
        default="hf-internal-testing/tiny-random-wav2vec2",
        type=str
    )

    parser.add_argument(
        "--device",
        default=-1,
        type=int
    )

    parser.add_argument(
        "--dataset_path",
        default="hf-internal-testing/librispeech_asr_dummy",
        type=str
    )
    parser.add_argument(
        "--dataset_name",
        default="clean",
        type=str
    )
    parser.add_argument(
        "--dataset_split",
        default="validation[:10]",
        type=str
    )
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    pipe = pipeline(
        model=args.model_name,
        device=args.device
    )
    dataset = load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        split=args.dataset_split,
        cache_dir=args.dataset_cache_dir
    )

    for out in pipe(KeyDataset(dataset=dataset, key="audio")):
        print(out)
    return


if __name__ == '__main__':
    main()
