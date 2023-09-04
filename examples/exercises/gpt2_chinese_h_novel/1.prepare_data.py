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

    parser.add_argument(
        "--pretrained_model_name_or_path",
        default=(project_path / "pretrained_models/gpt2-chinese-cluecorpussmall").as_posix(),
        type=str
    )

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


class TextNormalization(object):
    """
    https://blog.csdn.net/buluxianfeng/article/details/126223346
    """

    def __init__(self):
        pass

    def is_q_number(self, uchar):
        """判断一个unicode是否是全角数字"""
        if u'\uff10' <= uchar <= u'\uff19':
            return True
        else:
            return False

    def is_q_alphabet(self, uchar):
        """判断一个unicode是否是全角英文字母"""
        if (u'\uff21' <= uchar <= u'\uff3a') or (u'\uff41' <= uchar <= u'\uff5a'):
            return True
        else:
            return False

    def q_to_b(self, uchar):
        """单个字符 全角转半角"""
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:
            return uchar
        return chr(inside_code)

    def number_alphabet_q_to_b(self, text: str):
        result = ""
        for c in text:
            if self.is_q_alphabet(c) or self.is_q_number(c):
                c = self.q_to_b(c)
            result += c
        return result

    def normalize(self, text: str):
        text = self.number_alphabet_q_to_b(text)
        return text


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

    text_normalize = TextNormalization()

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

            text = text_normalize.normalize(text)

            outputs = tokenizer.tokenize(text)
            if tokenizer.unk_token in outputs:
                print(text)
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
