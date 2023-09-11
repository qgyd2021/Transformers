#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/docs/transformers/main_classes/deepspeed
https://zhuanlan.zhihu.com/p/630734624
"""
import argparse
import os
import platform

import datasets
from datasets import Dataset, DatasetDict, load_dataset
from datasets.splits import NamedSplit
import sentencepiece
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import BloomTokenizerFast, BloomForCausalLM
from transformers import AutoTokenizer, AutoModel
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", default="Iess/chinese_modern_poetry", type=str)
    parser.add_argument("--dataset_name", default=None, type=str)
    parser.add_argument("--dataset_split", default=None, type=str)
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="Qwen/Qwen-7B",
        # default=(project_path / "pretrained_models/huggingface/bigscience/bloom-1b7").as_posix(),
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
    # parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16", action="store_false")
    parser.add_argument("--half_precision_backend", default="auto", type=str)
    parser.add_argument("--dataloader_num_workers", default=5, type=int)
    parser.add_argument("--disable_tqdm", action="store_false")
    parser.add_argument("--remove_unused_columns", action="store_false")
    parser.add_argument("--deepspeed", default="ds_z3_config.json", type=str)
    # parser.add_argument("--deepspeed", default=None, type=str)
    parser.add_argument("--optim", default="adamw_hf", type=str)
    parser.add_argument("--report_to", default="tensorboard", type=str)
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    # parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_false")

    parser.add_argument("--truncate_longer_samples", action="store_false")
    parser.add_argument("--max_seq_length", default=1024, type=int)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # dataset
    # train_files = [
    #     (project_path / "datasets/chinese_modern_poetry/training_imagery2_maxlen256.json").as_posix(),
    #     (project_path / "datasets/chinese_modern_poetry/training_imagery3_maxlen256.json").as_posix(),
    #     (project_path / "datasets/chinese_modern_poetry/training_imagery4_maxlen256.json").as_posix(),
    #     (project_path / "datasets/chinese_modern_poetry/training_imagery5_maxlen256.json").as_posix(),
    # ]
    # train_dataset = Dataset.from_json(path_or_paths=train_files, split=datasets.Split.TRAIN)

    dataset_dict = load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        split=args.dataset_split,
        cache_dir=args.dataset_cache_dir
    )
    train_dataset = dataset_dict["train"]
    print(train_dataset)

    # pretrained model
    # tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    # model = AutoModel.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer = BloomTokenizerFast.from_pretrained(args.pretrained_model_name_or_path)
    model = BloomForCausalLM.from_pretrained(args.pretrained_model_name_or_path)

    def encode_with_truncation(examples):
        prompt_ = examples.pop('prompt')
        response_ = examples.pop('response')
        text = '<s>{input}</s>{target}</s>'.format(input=prompt_, target=response_)
        result = tokenizer.__call__(
            text,
            truncation=True,
            # padding='max_length',
            max_length=args.max_seq_length,
            return_special_tokens_mask=True
        )
        return result

    train_dataset = train_dataset.map(
        encode_with_truncation,
        batched=False,
        keep_in_memory=False,
        num_proc=None if platform.system() == 'Windows' else os.cpu_count(),
        cache_file_name=os.path.join(args.cache_dir, 'train.cache')
    )
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    print('Train Dataset Examples Batch Number: {}'.format(len(train_dataset)))

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
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        no_cuda=args.no_cuda,
        fp16=args.fp16,
        half_precision_backend=args.half_precision_backend,
        dataloader_num_workers=args.dataloader_num_workers,
        disable_tqdm=args.disable_tqdm,
        remove_unused_columns=args.remove_unused_columns,
        deepspeed=args.deepspeed,
        optim=args.optim,
        report_to=args.report_to,
        resume_from_checkpoint=args.resume_from_checkpoint,
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
