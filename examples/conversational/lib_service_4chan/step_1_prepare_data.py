#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import platform

from datasets import load_dataset

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="qgyd2021/lip_service_4chan", type=str)
    parser.add_argument("--dataset_name", default="moss_003_sft_data_10", type=str)
    parser.add_argument("--dataset_split", default=None, type=str)
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )
    parser.add_argument("--dataset_streaming", default=False, type=bool)
    parser.add_argument(
        "--num_workers",
        default=None if platform.system() == "Windows" else os.cpu_count() // 2,
        type=str
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    dataset_dict = load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        split=args.dataset_split,
        cache_dir=args.dataset_cache_dir,
        num_proc=args.num_workers if not args.dataset_streaming else None,
        streaming=args.dataset_streaming,
    )
    print(dataset_dict)

    dataset = dataset_dict["train"]

    if args.dataset_streaming:
        valid_dataset = dataset.take(args.valid_dataset_size)
        train_dataset = dataset.skip(args.valid_dataset_size)
        train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer_size, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=10000, seed=None)
        train_dataset = dataset["train"]
        valid_dataset = dataset["test"]

    print(train_dataset)
    print(valid_dataset)
    return


if __name__ == '__main__':
    main()
