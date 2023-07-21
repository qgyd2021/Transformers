#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="bert-base-cased",
        type=str
    )
    parser.add_argument(
        "--question",
        default="How many programming languages does BLOOM support?",
        type=str
    )
    parser.add_argument(
        "--context",
        default="BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages.",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # fast
    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )
    inputs: BatchEncoding = tokenizer.__call__(
        args.question,
        args.context,
        max_length=32,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    print(inputs.keys())

    input_ids = inputs["input_ids"]
    offset_mapping = inputs["offset_mapping"]
    text = tokenizer.decode(input_ids)
    print(text)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(tokens)

    text_list = [args.question, args.context]
    text_idx = -1
    for offset in offset_mapping:
        begin = offset[0]
        end = offset[1]
        if begin == 0 and end == 0:
            text_idx += 1
            print("\n")
            continue
        piece = text_list[text_idx][begin: end]
        print(piece, end="\t")

    print(inputs.sequence_ids(0))
    return


if __name__ == '__main__':
    main()
