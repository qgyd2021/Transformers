#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="facebook/wav2vec2-base",
        type=str
    )
    parser.add_argument(
        "--text",
        default="Do not meddle in the affairs of wizards, for they are subtle and quick to anger.",
        type=str
    )
    parser.add_argument("--dataset_path", default="PolyAI/minds14", type=str)
    parser.add_argument("--dataset_name", default="en-US", type=str)
    parser.add_argument("--dataset_split", default="train", type=str)
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
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
    example = dataset[0]["audio"]
    print(example)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    example = dataset[0]["audio"]
    print(example)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )
    audio_input = [dataset[0]["audio"]["array"]]
    example = feature_extractor(audio_input, sampling_rate=16000)
    print(example)

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            padding=True,
            max_length=100000,
            truncation=True,
        )
        return inputs

    processed_dataset = preprocess_function(dataset[:5])

    example = processed_dataset["input_values"][0].shape
    print(example)

    return


if __name__ == '__main__':
    main()
