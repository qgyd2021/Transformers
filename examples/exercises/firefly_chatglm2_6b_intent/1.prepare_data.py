#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from tqdm import tqdm

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", default="qgyd2021/telemarketing_intent", type=str)
    parser.add_argument("--dataset_name", default="chinese_prompt", type=str)
    parser.add_argument("--dataset_split", default="train", type=str)
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

    dataset_dict = load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        # split=args.dataset_split,
        cache_dir=args.dataset_cache_dir,
    )
    print(dataset_dict)

    train_dataset = dataset_dict["train"]

    with open(args.train_subset, "w", encoding="utf-8") as ftrain, \
        open(args.valid_subset, "w", encoding="utf-8") as fvalid:
        for sample in tqdm(train_dataset):
            source = sample["source"]
            text = sample["text"]
            label0 = sample["label0"]
            label1 = sample["label1"]
            selected = sample["selected"]
            checked = sample["checked"]
            prompt = sample["prompt"]

            if source in ("download", ):
                continue
            if selected != 1:
                continue

            row = {
                "input": prompt,
                "target": label1
            }
            row = json.dumps(row, ensure_ascii=False)

            if random.random() < 0.95:
                ftrain.write("{}\n".format(row))
            else:
                fvalid.write("{}\n".format(row))

    return


if __name__ == '__main__':
    main()
