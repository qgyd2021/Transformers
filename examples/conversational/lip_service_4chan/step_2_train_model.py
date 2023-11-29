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

from datasets import load_dataset, concatenate_datasets
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
    dataset_path: str = field(default="qgyd2021/lip_service_4chan")
    dataset_name: str = field(default=None)
    dataset_split: str = field(default=None)
    dataset_cache_dir: str = field(default=(project_path / "hub_datasets").as_posix())
    dataset_streaming: bool = field(default=False)
    num_workers: int = field(default=None if platform.system() == "Windows" else os.cpu_count() // 2)

    # model
    pretrained_model_name_or_path: str = field(
        default="uer/gpt2-chinese-cluecorpussmall"
    )
    # pretrained_model_name_or_path: str = field(
    #     default=(project_path / "pretrained_models/gpt2-chinese-cluecorpussmall").as_posix()
    # )

    hf_token: str = field(default=None)


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
    name_list = [
        "chatterbot_10",
        "moss_003_sft_data_10",
        "weibo_1",
        "xiaohuangji_10",
    ]

    dataset = list()
    for name in name_list:
        dataset_dict = load_dataset(
            path=args.dataset_path,
            name=name,
            split=args.dataset_split,
            cache_dir=args.dataset_cache_dir,
            # num_proc=args.num_workers if not args.dataset_streaming else None,
            streaming=args.dataset_streaming,
        )
        # print(dataset_dict)
        dataset.append(dataset_dict["train"])
    dataset = concatenate_datasets(dataset)

    if args.dataset_streaming:
        valid_dataset = dataset.take(args.valid_dataset_size)
        train_dataset = dataset.skip(args.valid_dataset_size)
        train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer_size, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=4000, seed=None)
        train_dataset = dataset["train"]
        valid_dataset = dataset["test"]

    # pretrained model
    model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    # map
    def encode(examples: dict):
        questions_ = examples.pop("question")
        answers_ = examples.pop("answer")

        utterances = list()
        for question, answer in zip(questions_, answers_):
            if not isinstance(question, str):
                continue
            if not isinstance(answer, str):
                continue
            utterance = question + tokenizer.sep_token + answer
            utterances.append(utterance)

        utterances = tokenizer.__call__(
            text=utterances,
            truncation=True,
            padding="longest",
            max_length=512,
            return_special_tokens_mask=True,
        )
        return utterances

    train_dataset = train_dataset.map(
        encode,
        batched=True,
        drop_last_batch=True,
        batch_size=10,
        num_proc=None,
        cache_file_name="train.cache"
    )
    valid_dataset = valid_dataset.map(
        encode,
        batched=True,
        drop_last_batch=True,
        batch_size=10,
        num_proc=None,
        cache_file_name="valid.cache"
    )
    dataset_info = f"""
    train dataset: {len(train_dataset)}
    valid dataset: {len(valid_dataset)}
    """
    dataset_info = re.sub(r"[\u0020]{4,}", "", dataset_info)
    print(dataset_info)

    # for k, v in model.named_parameters():
    #     if k.__contains__(".bias"):
    #         v.requires_grad = True
    #     else:
    #         v.requires_grad = False

    # for k, v in model.named_parameters():
    #     if v.requires_grad is True:
    #         print(k)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # training_args
    training_args = TrainingArguments(
        output_dir="output_dir",
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0,
        max_grad_norm=1.0,
        num_train_epochs=1.0,
        warmup_steps=1000,
        logging_steps=100,
        save_strategy="steps",
        save_steps=100,
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
        hub_model_id="lib_service_4chan",
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
    train_on_kaggle_notebook()
