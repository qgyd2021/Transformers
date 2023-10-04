#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

from datasets import load_dataset

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", default="lvwerra/stack-exchange-paired", type=str)
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    train_dataset = load_dataset(
        path=args.dataset_path,
        data_dir="data/reward",
        split="train",
        cache_dir=args.dataset_cache_dir
    )
    eval_dataset = load_dataset(
        path=args.dataset_path,
        data_dir="data/evaluation",
        split="train",
        cache_dir=args.dataset_cache_dir
    )
    print(train_dataset)
    print(eval_dataset)

    return


if __name__ == '__main__':
    main()
