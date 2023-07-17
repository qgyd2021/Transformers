#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/docs/transformers/preprocessing
"""
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from datasets import load_dataset
from transformers import AutoImageProcessor
from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="google/vit-base-patch16-224",
        type=str
    )
    parser.add_argument(
        "--text",
        default="Do not meddle in the affairs of wizards, for they are subtle and quick to anger.",
        type=str
    )

    parser.add_argument("--dataset_path", default="food101", type=str)
    parser.add_argument("--dataset_split", default="train[:100]", type=str)
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
        split=args.dataset_split,
        cache_dir=args.dataset_cache_dir,
    )

    example = dataset[0]["image"]
    print(example)

    image_processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )

    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )

    _transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])

    def transforms(examples):
        images = [_transforms(img.convert("RGB")) for img in examples["image"]]
        examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
        return examples

    dataset.set_transform(transforms)

    keys = dataset[0].keys()
    print(keys)

    return


if __name__ == '__main__':
    main()
