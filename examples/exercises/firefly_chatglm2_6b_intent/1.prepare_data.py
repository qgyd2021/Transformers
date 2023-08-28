#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset

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

    return


if __name__ == '__main__':
    main()
