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
        "--train_file",
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

    parser.add_argument("--output_dir", default="serialization_dir", type=str)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--evaluation_strategy", default="no", choices=["no", "steps", "epoch"], type=str)
    parser.add_argument("--per_device_train_batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=1.0, type=float)
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
    parser.add_argument("--deepspeed", default="ds_z3_config.json", type=str)
    parser.add_argument("--optim", default="adamw_hf", type=str)
    parser.add_argument("--report_to", default="tensorboard", type=str)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--truncate_longer_samples", action="store_true")
    parser.add_argument("--max_seq_length", default=512, type=int)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # dataset
    dataset_dict = DatasetDict()
    train_data_files = [args.train_file]
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
            max_length=args.max_seq_length,
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
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        evaluation_strategy=args.evaluation_strategy,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        no_cuda=args.no_cuda,
        fp16=args.fp16,
        half_precision_backend=args.half_precision_backend,
        deepspeed=args.deepspeed,
        report_to=args.report_to,
        gradient_checkpointing=args.gradient_checkpointing,
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
