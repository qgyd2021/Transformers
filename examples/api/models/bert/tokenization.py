#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from transformers.models.bert.tokenization_bert import BertTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="bert-base-cased",
        type=str
    )
    parser.add_argument(
        "--text",
        default="Do not meddle in the affairs of wizards, for they are subtle and quick to anger.",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )
    result = tokenizer(args.text)
    print(result)

    batch_sentences = [
        "But what about second breakfast?",
        "Don't think he knows about second breakfast, Pip.",
        "What about elevensies?",
    ]
    result = tokenizer(
        batch_sentences,
        padding="max_length",
        max_length=12,
        truncation=True,
    )
    print(result)
    return


if __name__ == '__main__':
    main()
