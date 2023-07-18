#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/docs/transformers/tasks/question_answering
"""
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

import huggingface_hub
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering

import project_settings as settings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="qgyd2021/distilbert-base-uncased-squad-question-answering",
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
    inputs = tokenizer.__call__(args.text, return_tensors="pt")

    model = AutoModelForQuestionAnswering.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
    result = tokenizer.decode(predict_answer_tokens)
    print(result)
    return


if __name__ == '__main__':
    main()
