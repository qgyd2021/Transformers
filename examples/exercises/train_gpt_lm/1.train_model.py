#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
参考链接:
https://www.thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python

"""
import argparse
from itertools import chain
import os
from pathlib import Path
import platform

from datasets import Dataset, DatasetDict, load_dataset
import torch
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_subset", default="file_dir/train.txt", type=str)

    parser.add_argument(
        "--pretrained_model_dir",
        default=(project_path / "pretrained_models/gpt2-chinese-cluecorpussmall").as_posix(),
        type=str
    )

    parser.add_argument("--dataset_path", default="wybxc/books", type=str)
    parser.add_argument("--dataset_split", default=None, type=str)
    # parser.add_argument("--dataset_split", default="train[:5000]", type=str)
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )

    parser.add_argument("--truncate_longer_samples", action="store_true")
    parser.add_argument("--max_length", default=1024, type=int)
    parser.add_argument("--batch_size", default=4, type=int)

    parser.add_argument("--cache_dir", default="file_dir/cache", type=str)
    parser.add_argument("--output_dir", default=None, type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    dataset_dict = load_dataset(
        args.dataset_path,
        split=args.dataset_split,
        cache_dir=args.dataset_cache_dir,
    )

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_dir)
    model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_dir)

    def encode_with_truncation(examples):
        outputs = tokenizer.__call__(examples['text'],
                                     truncation=True,
                                     padding='max_length',
                                     max_length=args.max_length,
                                     return_special_tokens_mask=True)
        return outputs

    def encode_without_truncation(examples):
        outputs = tokenizer.__call__(examples['text'],
                                     return_special_tokens_mask=True)
        return outputs

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= args.max_length:
            total_length = (total_length // args.max_length) * args.max_length
        else:
            raise AssertionError("total_length: {} < args.max_length: {}".format(args.max_length, args.max_length))
        result = {
            k: [t[i: i + args.max_length] for i in range(0, total_length, args.max_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    if args.truncate_longer_samples:
        dataset_dict = dataset_dict.map(
            encode_with_truncation,
            batched=True,
            keep_in_memory=False,
            num_proc=None if platform.system() == 'Windows' else os.cpu_count(),
            # cache_file_names={
            #     "train": os.path.join(args.cache_dir, "train_tokenized.cache")
            # }
        )
        dataset_dict.set_format(type="torch", columns=["input_ids", "attention_mask"])
    else:
        dataset_dict = dataset_dict.map(
            encode_without_truncation,
            batched=True,
            keep_in_memory=False,
            num_proc=None if platform.system() == 'Windows' else os.cpu_count(),
            # cache_file_names={
            #     "train": os.path.join(args.cache_dir, "train_tokenized.cache")
            # }
        )
        dataset_dict.set_format(type="torch", columns=["input_ids", "attention_mask"])

        dataset_dict = dataset_dict.map(
            group_texts,
            batched=True,
            desc="Grouping texts in chunks of {}".format(args.max_length),
            keep_in_memory=False,
            num_proc=None if platform.system() == 'Windows' else os.cpu_count(),
            # cache_file_names={
            #     "train": os.path.join(args.cache_dir, "train.cache")
            # },
        )
        dataset_dict.set_format("torch")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="no",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=16,
        logging_steps=1000,
        save_steps=1000,
        fp16=True if torch.cuda.is_available() else False,
        local_rank=-1,
        # load_best_model_at_end=True,
        save_total_limit=5,
        report_to="tensorboard"
    )

    training_info = """
    TRAINING_INFO:    
 
    n_gpu: {n_gpu}
    parallel_mode: {parallel_mode}
    world_size: {world_size}
    device: {device}
    """.format(
        n_gpu=training_args.n_gpu,
        parallel_mode=training_args.parallel_mode,
        world_size=training_args.world_size,
        device=training_args.device,
    )
    print(training_info)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_dict["train"],
    )
    trainer.train()
    return


if __name__ == '__main__':
    main()
