#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.models.bert.tokenization_bert import BertTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="bert-base-cased",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )

    batch_sentences = [
        "But what about second breakfast?",
        "Don't think he knows about second breakfast, Pip.",
        "What about elevensies?",
    ]
    mapping = [{"text": d} for d in batch_sentences]
    dataset = Dataset.from_list(mapping)

    def preprocess_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
        )
        return result

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(column_names=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=3,
        collate_fn=data_collator
    )

    for batch in data_loader:
        print(batch)

    return


if __name__ == '__main__':
    main()
