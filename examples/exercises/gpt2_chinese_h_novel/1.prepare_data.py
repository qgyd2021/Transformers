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
from transformers.models.bert.tokenization_bert import BertTokenizer

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", default="qgyd2021/h_novel", type=str)
    # parser.add_argument("--dataset_name", default="ltxsba_500m", type=str)
    parser.add_argument("--dataset_name", default="ltxsba_5gb", type=str)
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
        streaming=True,
    )

    train_dataset = dataset_dict["train"]

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    with open(args.train_subset, "w", encoding="utf-8") as ftrain, \
        open(args.valid_subset, "w", encoding="utf-8") as fvalid:
        for sample in tqdm(train_dataset):
            # print(sample)

            source = sample["source"]
            idx = sample["idx"]
            filename = sample["filename"]
            novel_name = sample["novel_name"]
            row_idx = sample["row_idx"]
            text = sample["text"]

            outputs = tokenizer.tokenize(text)
            print(outputs)
            exit(0)
            row = {
                "text": text
            }
            row = json.dumps(row, ensure_ascii=False)

            if random.random() < 0.95:
                ftrain.write("{}\n".format(row))
            else:
                fvalid.write("{}\n".format(row))

    return


if __name__ == '__main__':
    main()
