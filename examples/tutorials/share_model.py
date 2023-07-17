#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/docs/transformers/model_sharing
"""
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from accelerate import Accelerator
from datasets import load_dataset
import evaluate
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from transformers.models.auto.modeling_auto import AutoModel
import huggingface_hub

import project_settings as settings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="openai/whisper-tiny",
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

    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
    )

    huggingface_hub.login(token=args.hf_token)
    model.push_to_hub("qgyd2021/whisper-tiny")

    return


if __name__ == '__main__':
    main()
