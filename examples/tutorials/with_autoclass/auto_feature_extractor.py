#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from transformers import AutoFeatureExtractor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="google/vit-base-patch16-224",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )
    print(feature_extractor)
    return


if __name__ == '__main__':
    main()
