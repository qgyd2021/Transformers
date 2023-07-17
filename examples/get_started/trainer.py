#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/docs/transformers/quicktour#trainer-a-pytorch-optimized-training-loop
"""
import argparse
import os
from typing import List

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="distilbert-base-uncased",
        type=str
    )
    parser.add_argument(
        "--dataset_path",
        default="rotten_tomatoes",
        type=str
    )
    parser.add_argument("--output_dir", default="output_dir", type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    dataset = load_dataset(args.dataset_path)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenize_dataset(dataset):
        return tokenizer(dataset["text"])

    dataset = dataset.map(tokenize_dataset, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    return


if __name__ == '__main__':
    main()
