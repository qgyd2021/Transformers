#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://github.com/thunlp/OpenDelta/blob/main/examples/tutorial/1_with_openprompt.py
"""
import argparse
from collections import defaultdict
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from datasets import load_dataset
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="super_glue", type=str)
    parser.add_argument("--dataset_name", default="cb", type=str)
    parser.add_argument("--dataset_split", default=None, type=str)
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # dataset
    raw_dataset_dict = load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        split=args.dataset_split,
        cache_dir=args.dataset_cache_dir,
    )
    print(raw_dataset_dict)

    dataset = defaultdict(list)
    for split, raw_dataset in raw_dataset_dict.items():
        for sample in raw_dataset:
            input_example = InputExample(
                text_a=sample['premise'],
                text_b=sample['hypothesis'],
                label=int(sample['label']),
                guid=sample['idx']
            )
            dataset[split].append(input_example)

    print(dataset['train'][0])

    plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")
    template_text = '{"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} {"placeholder":"text_b"}? {"soft"} {"soft"} {"soft"} {"mask"}.'
    mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text=template_text)
    print(mytemplate)

    wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
    print(wrapped_example)
    wrapped_t5tokenizer = WrapperClass(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer, truncate_method="head")

    return


if __name__ == '__main__':
    main()
