#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
参考链接:
https://www.thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python
https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

"""
import argparse
from itertools import chain
import os
from pathlib import Path
import platform

from datasets import Dataset, DatasetDict, IterableDatasetDict, IterableDataset, load_dataset
import torch
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default=(project_path / "pretrained_models/gpt2-chinese-cluecorpussmall").as_posix(),
        type=str
    )

    parser.add_argument("--train_subset", default="train.jsonl", type=str)
    parser.add_argument("--valid_subset", default="valid.jsonl", type=str)

    parser.add_argument("--output_dir", default="serialization_dir", type=str)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--evaluation_strategy", default="no", choices=["no", "steps", "epoch"], type=str)
    parser.add_argument("--per_device_train_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    # parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--max_steps", default=3e8, type=int)
    parser.add_argument("--lr_scheduler_type", default="cosine", type=str)
    parser.add_argument("--warmup_ratio", default=0.0, type=float)
    parser.add_argument("--warmup_steps", default=3000, type=int)
    parser.add_argument("--logging_steps", default=300, type=int)
    parser.add_argument("--save_strategy", default="steps", type=str)
    parser.add_argument("--save_steps", default=500, type=int)
    parser.add_argument("--save_total_limit", default=3, type=int)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", default=3407, type=str, help="https://arxiv.org/abs/2109.08203")
    # parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16", action="store_false")
    parser.add_argument("--half_precision_backend", default="auto", type=str)
    parser.add_argument("--dataloader_num_workers", default=5, type=int)
    parser.add_argument("--disable_tqdm", action="store_false")
    parser.add_argument("--remove_unused_columns", action="store_false")
    # parser.add_argument("--deepspeed", default="ds_z3_config.json", type=str)
    parser.add_argument("--deepspeed", default=None, type=str)
    parser.add_argument("--optim", default="adamw_hf", type=str)
    parser.add_argument("--report_to", default="tensorboard", type=str)
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    # parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_false")

    parser.add_argument("--truncate_longer_samples", action="store_true")
    # parser.add_argument("--truncate_longer_samples", action="store_false")
    parser.add_argument("--max_seq_length", default=1024, type=int)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # dataset
    # dataset_dict = DatasetDict()
    dataset_dict = IterableDatasetDict()
    train_data_files = [args.train_subset]
    dataset_dict["train"] = load_dataset(
        path="json", data_files=[str(file) for file in train_data_files],
        streaming=True,
    )["train"]
    valid_data_files = [args.valid_subset]
    dataset_dict["valid"] = load_dataset(
        path="json", data_files=[str(file) for file in valid_data_files],
        streaming=True,
    )["train"]

    print(dataset_dict)

    # model
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_name_or_path)

    def encode_with_truncation(examples):
        outputs = tokenizer.__call__(examples['text'],
                                     truncation=True,
                                     padding='max_length',
                                     max_length=args.max_seq_length,
                                     return_special_tokens_mask=True)
        return outputs

    def encode_without_truncation(examples):
        outputs = tokenizer.__call__(examples['text'],
                                     return_special_tokens_mask=True)
        return outputs

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= args.max_seq_length:
            total_length = (total_length // args.max_seq_length) * args.max_seq_length

            result = {
                k: [t[i: i + args.max_seq_length] for i in range(0, total_length, args.max_seq_length)]
                for k, t in concatenated_examples.items()
            }
        else:
            result = {
                k: [] for k, t in concatenated_examples.items()
            }

        return result

    if args.truncate_longer_samples:
        dataset_dict = dataset_dict.map(
            encode_with_truncation,
            batched=True,
            drop_last_batch=True,
            # keep_in_memory=False,
            # num_proc=None if platform.system() == 'Windows' else os.cpu_count() // 2,
            # num_proc=None,
        )
        # dataset_dict.set_format(type="torch", columns=["input_ids", "attention_mask"])
    else:
        dataset_dict = dataset_dict.map(
            encode_without_truncation,
            batched=True,
            drop_last_batch=True,
            # keep_in_memory=False,
            # num_proc=None if platform.system() == 'Windows' else os.cpu_count() // 2,
            # num_proc=None,
        )
        # dataset_dict.set_format(type="torch", columns=["input_ids", "attention_mask"])

        dataset_dict = dataset_dict.map(
            group_texts,
            batched=True,
            drop_last_batch=True,
            # keep_in_memory=False,
            # num_proc=None if platform.system() == 'Windows' else os.cpu_count() // 2,
            # num_proc=None,
        )
        # dataset_dict.set_format("torch")

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
        # deepspeed=args.deepspeed,
        report_to=args.report_to,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_dict["train"],
    )
    train_result = trainer.train()

    # 保存最好的 checkpoint
    final_save_path = os.path.join(training_args.output_dir, "final")
    trainer.save_model(final_save_path)  # Saves the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    tokenizer.save_pretrained(final_save_path)
    return


if __name__ == '__main__':
    main()
