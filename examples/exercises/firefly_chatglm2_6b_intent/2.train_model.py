#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from dataclasses import dataclass, field
import os
import platform
import sys
from typing import Optional

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

from datasets import Dataset, DatasetDict, load_dataset
import torch
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import BloomTokenizerFast, BloomForCausalLM
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer

from toolbox.transformers.data.dataset.dataset import SFTDataset, ChatGLM2SFTDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_subset", default="train.jsonl", type=str)
    parser.add_argument("--valid_subset", default="valid.jsonl", type=str)

    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="YeungNLP/firefly-chatglm2-6b",
        type=str
    )

    parser.add_argument("--cache_dir", default="cache_dir", type=str)

    parser.add_argument("--output_dir", default="serialization_dir", type=str)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--evaluation_strategy", default="no", choices=["no", "steps", "epoch"], type=str)
    parser.add_argument("--per_device_train_batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--lr_scheduler_type", default="cosine", type=str)
    parser.add_argument("--warmup_ratio", default=0.0, type=float)
    parser.add_argument("--warmup_steps", default=3000, type=int)
    parser.add_argument("--logging_steps", default=300, type=int)
    parser.add_argument("--save_strategy", default="steps", type=str)
    parser.add_argument("--save_steps", default=500, type=int)
    parser.add_argument("--save_total_limit", default=3, type=int)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", default=3407, type=str, help="https://arxiv.org/abs/2109.08203")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--half_precision_backend", default="auto", type=str)
    parser.add_argument("--dataloader_num_workers", default=5, type=int)
    parser.add_argument("--disable_tqdm", action="store_false")
    parser.add_argument("--remove_unused_columns", action="store_false")
    # parser.add_argument("--deepspeed", default="ds_z3_config.json", type=str)
    parser.add_argument("--deepspeed", default=None, type=str)
    parser.add_argument("--optim", default="adamw_hf", type=str)
    parser.add_argument("--report_to", default="tensorboard", type=str)
    parser.add_argument("--resume_from_checkpoint", default="file_dir/serialization_dir/checkpoint-103000", type=str)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    # parser.add_argument("--gradient_checkpointing", action="store_false")

    parser.add_argument("--truncate_longer_samples", action="store_true")
    parser.add_argument("--max_seq_length", default=512, type=int)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        trust_remote_code=True,
        use_fast=False if model.config.model_type == "llama" else True
    )

    # dataset
    train_dataset = ChatGLM2SFTDataset(tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    train_dataset.read(args.train_subset)
    valid_dataset = ChatGLM2SFTDataset(tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    valid_dataset.read(args.valid_subset)

    for sample in valid_dataset:
        print(sample)

    return


if __name__ == '__main__':
    main()
