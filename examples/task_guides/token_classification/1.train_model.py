#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/docs/transformers/tasks/token_classification

"""
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from datasets import load_dataset
import evaluate
import huggingface_hub
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers.tokenization_utils_base import BatchEncoding

import project_settings as settings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="distilbert-base-uncased",
        type=str
    )
    parser.add_argument(
        "--text",
        default="Do not meddle in the affairs of wizards, for they are subtle and quick to anger.",
        type=str
    )
    parser.add_argument("--dataset_path", default="wnut_17", type=str)
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )

    parser.add_argument(
        "--file_dir",
        default="file_dir",
        type=str
    )

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
    )
    parser.add_argument(
        "--hub_model_id",
        default="distilbert-base-uncased-wnut17-ner",
        type=str
    )
    parser.add_argument(
        "--hub_strategy",
        default="end",
        choices=["end", "every_save", "checkpoint", "all_checkpoints"],
        type=str
    )
    parser.add_argument(
        "--hf_token",
        default=settings.environment.get("hf_token", default=None, dtype=str),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    huggingface_hub.login(token=args.hf_token)

    # dataset
    dataset = load_dataset(
        path=args.dataset_path,
        cache_dir=args.dataset_cache_dir,
    )
    print((dataset["train"][0]))

    label_list = dataset["train"].features[f"ner_tags"].feature.names
    print(label_list)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )

    example = dataset["train"][0]
    tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
    print(tokens)

    def tokenize_and_align_labels(examples):
        tokenized_inputs: BatchEncoding = tokenizer.__call__(
            text=examples["tokens"],
            truncation=True,
            is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    print((tokenized_dataset["train"][0]))

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    id2label = {
        0: "O",
        1: "B-corporation",
        2: "I-corporation",
        3: "B-creative-work",
        4: "I-creative-work",
        5: "B-group",
        6: "I-group",
        7: "B-location",
        8: "I-location",
        9: "B-person",
        10: "I-person",
        11: "B-product",
        12: "I-product",
    }
    label2id = {
        "O": 0,
        "B-corporation": 1,
        "I-corporation": 2,
        "B-creative-work": 3,
        "I-creative-work": 4,
        "B-group": 5,
        "I-group": 6,
        "B-location": 7,
        "I-location": 8,
        "B-person": 9,
        "I-person": 10,
        "B-product": 11,
        "I-product": 12,
    }

    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        num_labels=13,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=args.file_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy=args.hub_strategy,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.push_to_hub()

    return


if __name__ == '__main__':
    main()
