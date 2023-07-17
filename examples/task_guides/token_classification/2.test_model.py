#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/docs/transformers/tasks/sequence_classification
https://huggingface.co/datasets/imdb
"""
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

import huggingface_hub
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification

import project_settings as settings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="qgyd2021/distilbert-base-uncased-imdb-classification",
        type=str
    )
    parser.add_argument(
        "--text",
        default="Do not meddle in the affairs of wizards, for they are subtle and quick to anger.",
        type=str
    )
    parser.add_argument(
        "--hf_token",
        default=settings.environment.get("hf_token", default=None, dtype=str),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    huggingface_hub.login(token=args.hf_token)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )
    inputs = tokenizer(args.text, return_tensors="pt")

    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
    print(predicted_token_class)

    return


if __name__ == '__main__':
    main()
