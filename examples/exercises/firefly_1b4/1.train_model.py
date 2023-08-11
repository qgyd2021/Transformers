#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import platform

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from datasets import Dataset, DatasetDict, load_dataset
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import BloomTokenizerFast, BloomForCausalLM
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_file",
        # default='firefly-train-1.1M.jsonl',
        default="D:/programmer/nlp_datasets/firefly-train-1.1M.jsonl",
        type=str
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        # default='YeungNLP/bloom-1b4-zh',
        default="D:/programmer/nlp_pretrained_model/bloom-1b7",
        type=str,
    )
    parser.add_argument("--cache_dir", default="cache_dir", type=str)

    parser.add_argument("--serialization_dir", default="serialization_dir", type=str)
    parser.add_argument("--evaluation_strategy", default="no", choices=["no", "steps", "epoch"], type=str)

    parser.add_argument("--truncate_longer_samples", action="store_true")
    parser.add_argument("--max_length", default=512, type=int)

    parser.add_argument("--per_device_train_batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)

    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)

    parser.add_argument("--no_cuda", action="store_true",
                        help="when use `--no_cuda` in command line, the value is True otherwise False")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--half_precision_backend", default="auto", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    os.makedirs(args.serialization_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # dataset
    dataset_dict = DatasetDict()
    train_data_files = [args.corpus_file]
    dataset_dict["train"] = load_dataset(
        path="json", data_files=[str(file) for file in train_data_files]
    )["train"]
    print(dataset_dict)

    # pretrained model
    tokenizer = BloomTokenizerFast.from_pretrained(args.pretrained_model_name_or_path)
    model = BloomForCausalLM.from_pretrained(args.pretrained_model_name_or_path)

    def encode_with_truncation(examples):
        input_ = examples.pop("input")
        target_ = examples.pop("target")
        text = "<s>{input}</s></s>{target}</s>".format(input=input_, target=target_)
        result = tokenizer(
            text,
            truncation=True,
            # padding='max_length',
            max_length=args.max_length,
            return_special_tokens_mask=True

        )
        return result

    train_dataset = dataset_dict["train"].map(
        encode_with_truncation,
        batched=False,
        keep_in_memory=False,
        num_proc=None if platform.system() == "Windows" else os.cpu_count(),
        cache_file_name=os.path.join(args.cache_dir, "train.cache")
    )
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    print("Train Dataset Examples Batch Number: {}".format(len(train_dataset)))

    # training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    training_args = TrainingArguments(
        output_dir=args.serialization_dir,
        overwrite_output_dir=True,
        evaluation_strategy="no",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=1000,
        save_steps=1000,
        save_total_limit=5,
        no_cuda=args.no_cuda,
        fp16=args.fp16,
        half_precision_backend=args.half_precision_backend,
        deepspeed={

        }
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    return


if __name__ == '__main__':
    main()
