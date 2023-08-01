#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

from transformers.models.bert.tokenization_bert import BertTokenizer

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--text", default="没有共产党就没有新中国. ", type=str)
    parser.add_argument(
        "--pretrained_model_dir",
        default=(project_path / "pretrained_models/chinese-bert-wwm-ext").as_posix(),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_dir)

    outputs = tokenizer.__call__(
        text=["没有共产党就没有新中国. ", "人有多大胆, 地有多大产. 肥猪赛大象. "],
        return_special_tokens_mask=True,
    )
    print(outputs)

    return


if __name__ == '__main__':
    main()
