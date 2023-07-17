#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/docs/transformers/pipeline_tutorial
"""
import argparse
import os

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
        "--audio",
        default="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    generator = pipeline(task=args.task)
    result: dict = generator(inputs=args.audio)
    print(result)
    return


if __name__ == '__main__':
    main()
