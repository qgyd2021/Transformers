#!/usr/bin/python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import os
from pathlib import Path
import platform
import re
from typing import Dict, List, Optional, Union

if platform.system() == "Windows":
    from project_settings import project_path
else:
    project_path = os.path.abspath("./")
    project_path = Path(project_path)

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from datasets import concatenate_datasets, load_dataset
import huggingface_hub
import torch
import torch.multiprocessing as mp
from transformers import HfArgumentParser
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.training_args import TrainingArguments


@dataclass
class ScriptArguments:
    # dataset
    dataset_path: str = field(default="qgyd2021/chinese_chitchat")
    dataset_name: str = field(default=None)
    dataset_split: str = field(default=None)
    dataset_cache_dir: str = field(default=(project_path / "hub_datasets").as_posix())
    dataset_streaming: bool = field(default=False)
    num_workers: int = field(default=None if platform.system() == "Windows" else os.cpu_count() // 2)

    valid_dataset_size: int = field(default=10000)
    seed: int = field(default=3407)

    # model
    # pretrained_model_name_or_path: str = field(
    #     default="uer/gpt2-chinese-cluecorpussmall" if platform.system() != "Windows" else (project_path / "pretrained_models/gpt2-chinese-cluecorpussmall").as_posix()
    # )
    pretrained_model_name_or_path: str = field(
        default="qgyd2021/chinese_chitchat"
    )
    hf_token: str = field(default="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")


def get_args():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    return args


def train_model(local_rank, world_size, args):
    os.environ["RANK"] = f"{local_rank}"
    os.environ["LOCAL_RANK"] = f"{local_rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    huggingface_hub.login(token=args.hf_token)

    # dataset
    names = [
        # "qingyun", "chatterbot",
        # "douban", "ptt", "subtitle", "tieba", "weibo",
        "xiaohuangji"
    ]
    dataset_list = list()
    for name in names:
        dataset_dict = load_dataset(
            path=args.dataset_path,
            name=name,
            split=args.dataset_split,
            cache_dir=args.dataset_cache_dir,
            # num_proc=args.num_workers if not args.dataset_streaming else None,
            streaming=args.dataset_streaming,
        )

        dataset = dataset_dict["train"]
        dataset_list.append(dataset)

    dataset = concatenate_datasets(dataset_list)

    if args.dataset_streaming:
        valid_dataset = dataset.take(args.valid_dataset_size)
        train_dataset = dataset.skip(args.valid_dataset_size)
        train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer_size, seed=args.seed)
    else:
        dataset = dataset.train_test_split(test_size=args.valid_dataset_size, seed=args.seed)
        train_dataset = dataset["train"]
        valid_dataset = dataset["test"]

    # pretrained model
    model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    # map
    def encode(examples: dict):
        conversation_ = examples.pop("conversation")

        utterances = list()
        for row_ in conversation_:
            message_ = row_["message"]
            utterance = tokenizer.sep_token.join(message_)
            utterances.append(utterance)

        utterances = tokenizer.__call__(
            text=utterances,
            truncation=True,
            padding="longest",
            max_length=1024,
            return_special_tokens_mask=True,
        )
        return utterances

    train_dataset = train_dataset.map(
        encode,
        batched=True,
        drop_last_batch=True,
        batch_size=10,
        num_proc=args.num_workers if not args.dataset_streaming else None,
        cache_file_name="train.cache"
    )
    valid_dataset = valid_dataset.map(
        encode,
        batched=True,
        drop_last_batch=True,
        batch_size=10,
        num_proc=args.num_workers if not args.dataset_streaming else None,
        cache_file_name="valid.cache"
    )
    dataset_info = f"""
    train dataset: {len(train_dataset)}
    valid dataset: {len(valid_dataset)}
    """
    dataset_info = re.sub(r"[\u0020]{4,}", "", dataset_info)
    print(dataset_info)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # training_args
    training_args = TrainingArguments(
        output_dir="output_dir",
        evaluation_strategy="steps",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0,
        max_grad_norm=1.0,
        num_train_epochs=40.0,
        warmup_steps=10000,
        logging_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        no_cuda=False,
        fp16=True if torch.cuda.is_available() else False,
        local_rank=local_rank,
        ddp_backend="nccl",
        remove_unused_columns=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="tensorboard",
        push_to_hub=True,
        hub_model_id="chinese_chitchat",
        hub_strategy="every_save",
        gradient_checkpointing=True,
    )

    partial_state_str = f"""
    distributed_type: {training_args.distributed_state.distributed_type}
    local_process_index: {training_args.distributed_state.local_process_index}
    num_processes: {training_args.distributed_state.num_processes}
    process_index: {training_args.distributed_state.process_index}
    device: {training_args.distributed_state.device}
    """
    partial_state_str = re.sub(r"[\u0020]{4,}", "", partial_state_str)
    print(partial_state_str)

    environ = f"""
    RANK: {os.environ.get("RANK", -1)}
    WORLD_SIZE: {os.environ.get("WORLD_SIZE", -1)}
    LOCAL_RANK: {os.environ.get("LOCAL_RANK", -1)}
    """
    environ = re.sub(r"[\u0020]{4,}", "", environ)
    print(environ)

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=5)
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks
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


def train_on_cpu():
    args = get_args()

    train_model(0, 1, args)
    return


def train_on_kaggle_notebook():
    """
    train on kaggle notebook with GPU T4 x2

    from shutil import copyfile
    copyfile(src = "../input/tempdataset/step_2_train_model.py", dst = "../working/step_2_train_model.py")

    import step_2_train_model
    step_2_train_model.train_on_kaggle_notebook()

    """
    args = get_args()

    world_size = torch.cuda.device_count()
    print("world_size: {}".format(world_size))

    mp.spawn(train_model,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

    return


if __name__ == '__main__':
    train_on_cpu()
