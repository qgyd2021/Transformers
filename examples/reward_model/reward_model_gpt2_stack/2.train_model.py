#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
reference:
https://github.com/huggingface/trl

https://huggingface.co/docs/trl/main/en/reward_trainer
https://huggingface.co/docs/trl/index
https://huggingface.co/blog/trl-peft

https://medium.com/towards-generative-ai/reward-model-training-2209d1befb5f

https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/reward_modeling.py
"""
import argparse
from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
from transformers.models.gpt2.modeling_gpt2 import GPT2ForSequenceClassification
from transformers.trainer_utils import EvalPrediction, IntervalStrategy

from project_settings import project_path


@dataclass
class ScriptArguments:
    # dataset
    dataset_path: str = field(default="lvwerra/stack-exchange-paired")
    dataset_cache_dir: str = field(default=(project_path / "hub_datasets").as_posix())
    train_subset: Optional[int] = field(default=-1)
    eval_subset: Optional[int] = field(default=10000)

    # cache
    cache_dir: str = field(default="cache_dir")

    # model
    model_name: Optional[str] = field(default="gpt2")
    num_labels: Optional[int] = field(default=1)
    last_checkpoint: Optional[str] = field(default="last_checkpoint")

    # tokenizer
    tokenizer_name: Optional[str] = field(default=None)

    # dataset process
    max_length: Optional[int] = field(default=512)

    # lora
    lora_rank: int = field(default=64)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)

    # training_args
    output_dir: Optional[str] = field(default="output_dir")
    evaluation_strategy: Union[IntervalStrategy, str] = field(default="steps")
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[float] = field(default=0.001)
    num_train_epochs: float = field(default=1.0)
    lr_scheduler_type: Optional[str] = field(default="linear")
    logging_strategy: Union[IntervalStrategy, str] = field(default="steps")
    save_strategy: Union[IntervalStrategy, str] = field(default="steps")
    logging_steps: float = field(default=500)
    bf16: bool = field(default=False)
    fp16: bool = field(default=False)
    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    eval_steps: Optional[float] = field(default=5000)
    save_steps: float = field(default=500)
    save_total_limit: Optional[int] = field(default=5)
    remove_unused_columns: Optional[bool] = field(default=False)
    label_names: Optional[List[str]] = field(default=None)
    deepspeed: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_hf")
    report_to: Optional[List[str]] = field(default=None)
    resume_from_checkpoint: Optional[bool] = field(default=False)
    gradient_checkpointing: Optional[bool] = field(default=False)

    # addition
    eval_first_step: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run eval after the first step"},
    )


def get_args():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = PaddingStrategy.MAX_LENGTH
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append({
                "input_ids": feature["input_ids_j"],
                "attention_mask": feature["attention_mask_j"],
            })
            features_k.append({
                "input_ids": feature["input_ids_k"],
                "attention_mask": feature["attention_mask_k"],
            })
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


class RewardTrainer(Trainer):
    # Define how to compute the reward loss.
    # We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = - nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


def main():
    args = get_args()

    # dataset
    train_dataset = load_dataset(
        path=args.dataset_path,
        data_dir="data/reward",
        split="train",
        cache_dir=args.dataset_cache_dir
    )
    if args.train_subset > 0:
        train_dataset = train_dataset.select(range(args.train_subset))
    eval_dataset = load_dataset(
        path=args.dataset_path,
        data_dir="data/evaluation",
        split="train",
        cache_dir=args.dataset_cache_dir
    )
    if args.eval_subset > 0:
        eval_dataset = eval_dataset.select(range(args.eval_subset))

    # training_args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        bf16=args.bf16,
        fp16=args.fp16,
        local_rank=args.local_rank,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=args.remove_unused_columns,
        label_names=list() if args.label_names is None else args.label_names,
        deepspeed=args.deepspeed,
        optim=args.optim,
        report_to=args.report_to,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # tokenizer
    tokenizer_name = args.tokenizer_name if args.tokenizer_name is not None else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)

    # model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Need to do this for gpt2, because it doesn't have an official pad token.
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = not args.gradient_checkpointing
    original_columns = train_dataset.column_names

    # Turn the dataset into pairs of post + summaries,
    # where text_j is the preferred question + answer and text_k is the other.
    # Then tokenize the dataset.
    def preprocess_function(examples):
        new_examples = {
            "input_ids_j": [],
            "attention_mask_j": [],
            "input_ids_k": [],
            "attention_mask_k": [],
        }
        for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
            tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_j,
                                    max_length=args.max_length, truncation=True)
            tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_k,
                                    max_length=args.max_length, truncation=True)

            new_examples["input_ids_j"].append(tokenized_j["input_ids"])
            new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
            new_examples["input_ids_k"].append(tokenized_k["input_ids"])
            new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

        return new_examples

    # preprocess the dataset and filter out QAs that are longer than script_args.max_length
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=os.cpu_count() // 2,
        remove_columns=original_columns,
        cache_file_name=os.path.join(args.cache_dir, 'train.cache')
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_j"]) <= args.max_length and len(x["input_ids_k"]) <= args.max_length,
        num_proc=os.cpu_count() // 2,
    )

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=os.cpu_count() // 2,
        remove_columns=original_columns,
        cache_file_name=os.path.join(args.cache_dir, 'train.cache')
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_j"]) <= args.max_length and len(x["input_ids_k"]) <= args.max_length,
        num_proc=os.cpu_count() // 2,
    )

    # Define the metric that we'll use for validation.
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, Any]:
        predictions, _ = eval_pred
        # Here, predictions is rewards_j and rewards_k.
        # We want to see how much of the time rewards_j > rewards_k.
        predictions = np.argmax(predictions, axis=0)
        labels = np.zeros(predictions.shape)
        return accuracy.compute(predictions=predictions, references=labels)

    # Train the model, woohoo.
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer,
                                                    padding="max_length",
                                                    max_length=args.max_length),
    )

    if args.eval_first_step:
        trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train(args.resume_from_checkpoint)

    print("Saving last checkpoint of the model")
    last_checkpoint = os.path.join(args.output_dir, args.last_checkpoint)
    model.save_pretrained(last_checkpoint)
    return


if __name__ == '__main__':
    main()
