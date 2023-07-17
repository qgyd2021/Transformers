#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/docs/transformers/quicktour
"""
import argparse
import os
from typing import List

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="sentiment-analysis",
        type=str
    )
    parser.add_argument(
        "--model_name",
        default="nlptown/bert-base-multilingual-uncased-sentiment",
        type=str
    )
    parser.add_argument(
        "--text1",
        default="Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.",
        type=str
    )
    parser.add_argument(
        "--text2",
        default="We are very happy to show you the 🤗 Transformers library.",
        type=str
    )
    parser.add_argument(
        "--text3",
        default="We hope you don't hate it.",
        type=str
    )

    parser.add_argument(
        "--save_directory",
        default=(project_path / "pretrained_models/huggingface").as_posix(),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    classifier = pipeline(
        task=args.task,
        model=model,
        tokenizer=tokenizer,
    )

    #
    results: List[dict] = classifier(args.text1)
    for result in results:
        print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

    #
    encoding = tokenizer(args.text2)
    print(encoding)

    #
    pt_batch = tokenizer(
        [args.text2, args.text3],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    print(pt_batch)

    pt_model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    pt_outputs = pt_model(**pt_batch)
    print(pt_outputs)

    pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
    print(pt_predictions)

    tokenizer.save_pretrained(os.path.join(args.save_directory, args.model_name))
    pt_model.save_pretrained(os.path.join(args.save_directory, args.model_name))

    return


if __name__ == '__main__':
    main()
