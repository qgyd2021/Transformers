#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/docs/transformers/model_sharing
"""
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

import huggingface_hub
from transformers.models.auto.modeling_auto import AutoModel

import project_settings as settings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="openai/whisper-tiny",
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

    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
    )

    huggingface_hub.login(token=args.hf_token)
    model.push_to_hub("qgyd2021/whisper-tiny")

    return


if __name__ == '__main__':
    main()
