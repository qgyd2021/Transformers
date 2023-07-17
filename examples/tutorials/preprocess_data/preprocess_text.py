#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from transformers import AutoTokenizer


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

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )

    encoded_input = tokenizer(args.text)
    print(encoded_input)

    result = tokenizer.decode(encoded_input["input_ids"])
    print(result)

    # batch
    batch_sentences = [
        "But what about second breakfast?",
        "Don't think he knows about second breakfast, Pip.",
        "What about elevensies?",
    ]
    encoded_input = tokenizer(batch_sentences)
    print(encoded_input)

    encoded_input = tokenizer(batch_sentences, padding=True)
    print(encoded_input)

    encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
    print(encoded_input)

    encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
    print(encoded_input)

    return


if __name__ == '__main__':
    main()
