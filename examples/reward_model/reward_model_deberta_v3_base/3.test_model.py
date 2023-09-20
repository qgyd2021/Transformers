#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2ForSequenceClassification
from transformers.models.deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="OpenAssistant/reward-model-deberta-v3-base",
        # default=(project_path / "trained_models/firefly_chatglm2_6b_intent").as_posix(),
        type=str
    )
    parser.add_argument("--question", default="Explain nuclear fusion like I am five", type=str)
    parser.add_argument(
        "--answer",
        default="Nuclear fusion is the process by which two or more protons and neutrons combine to form a single nucleus. It is a very important process in the universe, as it is the source of energy for stars and galaxies. Nuclear fusion is also a key process in the production of energy for nuclear power plants.",
        type=str
    )

    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    reward_name = "OpenAssistant/reward-model-deberta-v3-base"
    rank_model: DebertaV2ForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(reward_name)
    tokenizer = AutoTokenizer.from_pretrained(reward_name)

    rank_model.eval()

    inputs = tokenizer(args.question, args.answer, return_tensors='pt')
    score = rank_model(**inputs).logits[0].cpu().detach()
    print(score)

    return


if __name__ == '__main__':
    main()
