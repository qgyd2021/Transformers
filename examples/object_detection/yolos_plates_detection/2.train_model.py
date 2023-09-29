#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
reference:
https://huggingface.co/spaces/nickmuchi/license-plate-detection-with-YOLOS
https://huggingface.co/docs/transformers/tasks/object_detection
"""
import argparse
from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
from transformers.models.gpt2.modeling_gpt2 import GPT2ForSequenceClassification
from transformers.trainer_utils import EvalPrediction, IntervalStrategy

from project_settings import project_path


@dataclass
class ScriptArguments:
    # dataset
    # https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
    dataset_path: str = field(default="keremberke/license-plate-object-detection")
    dataset_name: str = field(default="full")
    dataset_cache_dir: str = field(default=(project_path / "hub_datasets").as_posix())


def get_args():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args


def main():
    args = get_args()

    # dataset
    train_dataset = load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        split="train",
        cache_dir=args.dataset_cache_dir
    )
    eval_dataset = load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        split="validation",
        cache_dir=args.dataset_cache_dir
    )

    for sample in train_dataset:
        print(sample)
    return


if __name__ == '__main__':
    main()
