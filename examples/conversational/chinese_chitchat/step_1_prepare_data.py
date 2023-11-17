#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from itertools import chain
import os
from pathlib import Path
import platform

if platform.system() == "Windows":
    from project_settings import project_path
else:
    project_path = os.path.abspath("./")
    project_path = Path(project_path)

from datasets import load_dataset, concatenate_datasets, IterableDataset, Dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="qgyd2021/chinese_chitchat", type=str)
    parser.add_argument("--dataset_split", default=None, type=str)
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )
    parser.add_argument("--dataset_streaming", default=False, type=bool)
    parser.add_argument("--valid_dataset_size", default=10000, type=int)

    parser.add_argument(
        "--num_workers",
        default=None if platform.system() == "Windows" else os.cpu_count() // 2,
        type=str
    )

    parser.add_argument("--seed", default=3407, type=str, help="https://arxiv.org/abs/2109.08203")

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    names = [
        "qingyun", "chatterbot", "douban", "ptt", "subtitle", "tieba", "weibo", "xiaohuangji"
    ]
    dataset_list = list()
    for name in names:
        dataset_dict = load_dataset(
            path=args.dataset_path,
            name=name,
            split=args.dataset_split,
            cache_dir=args.dataset_cache_dir,
            num_proc=args.num_workers if not args.dataset_streaming else None,
            streaming=args.dataset_streaming,
        )

        dataset = dataset_dict["train"]
        dataset_list.append(dataset)

    dataset = concatenate_datasets(dataset_list)

    if args.dataset_streaming:
        valid_dataset = dataset.take(args.valid_dataset_size)
        train_dataset = dataset.skip(args.valid_dataset_size)
        train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer_size, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=args.valid_dataset_size, seed=None)
        train_dataset = dataset["train"]
        valid_dataset = dataset["test"]

    print(train_dataset)
    print(valid_dataset)
    return


if __name__ == '__main__':
    main()
