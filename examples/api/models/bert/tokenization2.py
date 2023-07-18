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
        "--text",
        default="Do not meddle in the affairs of wizards, for they are subtle and quick to anger.",
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
    result = tokenizer.tokenize(args.text)
    print(result)

    added_vocab = tokenizer.get_added_vocab()
    print(added_vocab)

    bert_tokenizer = tokenizer.backend_tokenizer
    print(bert_tokenizer)

    tokenized_inputs: BatchEncoding = tokenizer.__call__(args.text)
    word_ids = tokenized_inputs.word_ids(batch_index=0)
    print(word_ids)

    return


if __name__ == '__main__':
    main()
