#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
参考链接:
https://www.thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python

"""
import argparse
from itertools import chain
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_subset", default="file_dir/train.txt", type=str)
    parser.add_argument("--valid_subset", default="file_dir/valid.txt", type=str)

    parser.add_argument(
        "--pretrained_model_dir",
        default=(project_path / "pretrained_models/gpt2-chinese-cluecorpussmall").as_posix(),
        type=str
    )
    parser.add_argument("--truncate_longer_samples", action="store_true")
    parser.add_argument("--max_length", default=1024, type=int)
    parser.add_argument("--batch_size", default=64, type=int)

    parser.add_argument("--output_dir", default=None, type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # dataset
    dataset_dict = DatasetDict({
        "train": load_dataset(
            "text", data_files=[str(file) for file in [args.train_subset]]
        )["train"],
        "valid": load_dataset(
            "text", data_files=[str(file) for file in [args.valid_subset]]
        )["train"]
    })

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
        result = {
            k: [t[i: i + args.max_length] for i in range(0, total_length, args.max_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    if args.truncate_longer_samples:
        train_dataset = dataset_dict["train"].map(encode_with_truncation, batched=True)
        test_dataset = dataset_dict["valid"].map(encode_with_truncation, batched=True)
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    else:
        train_dataset = dataset_dict["train"].map(encode_without_truncation, batched=True)
        test_dataset = dataset_dict["valid"].map(encode_without_truncation, batched=True)
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        train_dataset = train_dataset.map(
            group_texts, batched=True, desc="Grouping texts in chunks of {}".format(args.max_length)
        )
        test_dataset = test_dataset.map(
            group_texts, batched=True, desc="Grouping texts in chunks of {}".format(args.max_length)
        )
        train_dataset.set_format("torch")
        test_dataset.set_format("torch")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=64,
        logging_steps=1000,
        save_steps=1000,
        fp16=True,
        # load_best_model_at_end=True,
        save_total_limit=5,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
    return


if __name__ == '__main__':
    main()
