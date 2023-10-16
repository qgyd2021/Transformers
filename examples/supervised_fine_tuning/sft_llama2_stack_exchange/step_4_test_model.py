#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from dataclasses import dataclass, field
import os
import platform
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="./file_dir/serialization_dir", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model = model.merge_and_unload()

    return


if __name__ == '__main__':
    main()
