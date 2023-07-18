#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/docs/transformers/tasks/question_answering
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
from transformers import AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer
from transformers import DefaultDataCollator

import project_settings as settings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="distilbert-base-uncased",
        type=str
    )
    parser.add_argument("--dataset_path", default="squad", type=str)
    parser.add_argument("--dataset_split", default="train[:5000]", type=str)
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
        default="distilbert-base-uncased-squad-question-answering",
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

    dataset = load_dataset(
        path=args.dataset_path,
        split=args.dataset_split,
        cache_dir=args.dataset_cache_dir,
    )
    dataset = dataset.train_test_split(test_size=0.2)
    print(dataset["train"][0])

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

    data_collator = DefaultDataCollator()

    model = AutoModelForQuestionAnswering.from_pretrained(
        args.pretrained_model_name_or_path,
    )

    training_args = TrainingArguments(
        output_dir=args.file_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
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
    )

    trainer.train()
    trainer.push_to_hub()

    return


if __name__ == '__main__':
    main()
