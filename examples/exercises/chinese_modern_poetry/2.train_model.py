#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import os
import platform
import sys
from typing import Optional

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

import bitsandbytes as bnb
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig

from project_settings import project_path
from toolbox.transformers.data.dataset.dataset import SFTDataset, ChatGLM2SFTDataset
from toolbox.transformers.data.data_collator import SFTDataCollator
from toolbox.transformers.modules.loss import TargetLMLoss
from toolbox.transformers.trainer import LoRATrainer


def get_args():
    """
    python3 2.train_model.py --pretrained_model_name_or_path /data/tianxing/PycharmProjects/Transformers/pretrained_models/huggingface/YeungNLP/firefly-chatglm2-6b

    """
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
        type=str
    )
    parser.add_argument("--cache_dir", default="cache_dir", type=str)

    # train
    parser.add_argument("--output_dir", default="serialization_dir", type=str)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--evaluation_strategy", default="no", choices=["no", "steps", "epoch"], type=str)
    parser.add_argument("--per_device_train_batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--max_grad_norm", default=0.3, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--lr_scheduler_type", default="constant_with_warmup", type=str)
    parser.add_argument("--warmup_ratio", default=0.0, type=float)
    parser.add_argument("--warmup_steps", default=3000, type=int)
    parser.add_argument("--logging_steps", default=300, type=int)
    parser.add_argument("--save_strategy", default="steps", type=str)
    parser.add_argument("--save_steps", default=500, type=int)
    parser.add_argument("--save_total_limit", default=2, type=int)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", default=3407, type=str, help="https://arxiv.org/abs/2109.08203")
    # parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16", action="store_false")
    parser.add_argument("--half_precision_backend", default="auto", type=str)
    parser.add_argument("--dataloader_num_workers", default=0, type=int)
    parser.add_argument("--disable_tqdm", action="store_true")
    # parser.add_argument("--disable_tqdm", action="store_false")
    parser.add_argument("--remove_unused_columns", action="store_true")
    # parser.add_argument("--remove_unused_columns", action="store_false")
    # parser.add_argument("--deepspeed", default="ds_z3_config.json", type=str)
    parser.add_argument("--deepspeed", default=None, type=str)
    parser.add_argument("--optim", default="paged_adamw_32bit", type=str)
    parser.add_argument("--report_to", default="tensorboard", type=str)
    parser.add_argument("--resume_from_checkpoint", default="file_dir/serialization_dir/checkpoint-103000", type=str)
    # parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_false")

    # dataset process
    parser.add_argument("--truncate_longer_samples", action="store_true")
    parser.add_argument("--max_seq_length", default=1024, type=int)

    # lora
    parser.add_argument("--lora_rank", default=64, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=int)

    args = parser.parse_args()
    return args


def verify_model_dtype(model):
    """
    查看模型种各种类型的参数的情况
    """
    dtype2param_num = defaultdict(int)   # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)   # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)   # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)   # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)

    # 统计全部参数中, 各种类型参数分布.
    total = 0
    print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    print()
    # 统计可训练参数中, 各种类型参数分布.
    print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)


def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def main():
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # dataset
    dataset_dict = load_dataset(
        path=args.dataset_path,
        name=args.dataset_name,
        split=args.dataset_split,
        cache_dir=args.dataset_cache_dir
    )
    train_dataset = dataset_dict["train"]
    print(train_dataset)

    # training_args
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
        # deepspeed=args.deepspeed,
        optim=args.optim,
        report_to=args.report_to,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        device_map={"": 0},
        load_in_4bit=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        ),
    )
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        trust_remote_code=True,
        use_fast=False if model.config.model_type == "llama" else True
    )
    # QWenTokenizer比较特殊, pad_token_id, bos_token_id, eos_token_id 均 为None. eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == "QWenTokenizer":
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id

    # model
    # casts all the non int8 modules to full precision (fp32) for stability
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    print(f"memory footprint of model: {model.get_memory_footprint() / (1024*1024*1024)} GB")

    # 找到所有需要插入adapter的全连接层
    target_modules = find_all_linear_names(model)
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.config.torch_dtype = torch.float32

    # 查看模型种各种类型的参数的情况
    verify_model_dtype(model)

    # 初始化损失函数
    loss_func = TargetLMLoss(ignore_index=-100)

    data_collator = SFTDataCollator(tokenizer, args.max_seq_length)

    # dataset
    def encode_with_truncation(examples):
        prompt_ = examples.pop('prompt')
        response_ = examples.pop('response')
        utterances = [
            "<s>{input}</s>".format(input=prompt_),
            "{target}</s>".format(target=response_)
        ]

        utterances_ids = tokenizer(utterances, add_special_tokens=False).input_ids

        input_ids = list()
        target_mask = list()
        for i, utterances_id in enumerate(utterances_ids):
            input_ids += utterances_id
            if i % 2 == 0:
                target_mask += [0] * (len(utterances_id))
            else:
                input_ids += [tokenizer.eos_token_id]
                target_mask += [1] * (len(utterances_id) + 1)

        assert len(input_ids) == len(target_mask)

        input_ids = input_ids[:args.max_seq_length]
        target_mask = target_mask[:args.max_seq_length]
        attention_mask = [1] * len(input_ids)

        assert len(input_ids) == len(target_mask) == len(attention_mask)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask
        }
        return inputs

    train_dataset = train_dataset.map(
        encode_with_truncation,
        batched=False,
        keep_in_memory=False,
        num_proc=None if platform.system() == "Windows" else os.cpu_count(),
        cache_file_name=os.path.join(args.cache_dir, "train.cache")
    )
    train_dataset.set_format(type='torch', columns=["input_ids", "attention_mask", "target_mask"])
    print("Train Dataset Examples Batch Number: {}".format(len(train_dataset)))

    # 初始化 Trainer
    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # tokenizer=tokenizer,
        data_collator=data_collator,
        compute_loss=loss_func
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
    return


if __name__ == '__main__':
    main()
