#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json

from datasets import load_dataset
from tqdm import tqdm

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="qgyd2021/few_shot_intent_sft", type=str)
    parser.add_argument("--dataset_split", default=None, type=str)
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )
    parser.add_argument("--train_subset", default="train.jsonl", type=str)
    parser.add_argument("--valid_subset", default="valid.jsonl", type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    name_list = [
        "amazon_massive_intent_en_us_prompt",
        "amazon_massive_intent_zh_cn_prompt",
        "atis_intent_prompt",
        "banking77_prompt",
        "conv_intent_prompt",
        "intent_classification_prompt",
        "mobile_assistant_prompt",
        "mtop_intent_prompt",
        "snips_built_in_intents_prompt",
        "telemarketing_intent_en_prompt",
        "telemarketing_intent_cn_prompt",
        "vira_intents_prompt",
    ]

    with open(args.train_subset, "w", encoding="utf-8") as f:
        for name in name_list:
            dataset = load_dataset(
                path=args.dataset_path,
                name=name,
                split="train",
                cache_dir=args.dataset_cache_dir
            )

            for sample in tqdm(dataset):
                row = json.dumps(sample, ensure_ascii=False)
                f.write("{}\n".format(row))

    with open(args.valid_subset, "w", encoding="utf-8") as f:
        for name in name_list:
            dataset = load_dataset(
                path=args.dataset_path,
                name=name,
                split="test",
                cache_dir=args.dataset_cache_dir
            )

            for sample in tqdm(dataset):
                row = json.dumps(sample, ensure_ascii=False)
                f.write("{}\n".format(row))

    return


if __name__ == '__main__':
    main()
