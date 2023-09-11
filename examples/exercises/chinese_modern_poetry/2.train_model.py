#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/docs/transformers/main_classes/deepspeed
https://zhuanlan.zhihu.com/p/630734624

deepspeed --num_gpus=1 2.train_model.py
"""
import argparse
import os
import platform
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

import datasets
from datasets import Dataset, DatasetDict, load_dataset
from datasets.splits import NamedSplit
import deepspeed
import sentencepiece
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import BloomTokenizerFast, BloomForCausalLM
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation import GenerationConfig
# from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from project_settings import project_path
from toolbox.transformers.data.dataset.dataset import SFTDataset, ChatGLM2SFTDataset
from toolbox.transformers.data.data_collator import SFTDataCollator
from toolbox.transformers.modules.loss import TargetLMLoss
from toolbox.transformers.trainer import Trainer


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
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dataloader_num_workers", default=5, type=int)
    parser.add_argument("--disable_tqdm", action="store_false")
    parser.add_argument("--remove_unused_columns", action="store_false")
    parser.add_argument("--deepspeed", default="ds_z2_cpu_offload_config.json", type=str)
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
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        trust_remote_code=True,
    )
    # QWenTokenizer比较特殊, pad_token_id, bos_token_id, eos_token_id 均 为None. eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == "QWenTokenizer":
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
    )

    # def encode_with_truncation(examples):
    #     prompt_ = examples.pop('prompt')
    #     response_ = examples.pop('response')
    #     text = '<s>{input}</s>{target}</s>'.format(input=prompt_, target=response_)
    #     result = tokenizer.__call__(
    #         text,
    #         truncation=True,
    #         # padding='max_length',
    #         max_length=args.max_seq_length,
    #         return_special_tokens_mask=True
    #     )
    #     return result

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
        num_proc=None if platform.system() == 'Windows' else os.cpu_count(),
        cache_file_name=os.path.join(args.cache_dir, 'train.cache')
    )
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    print('Train Dataset Examples Batch Number: {}'.format(len(train_dataset)))

    # training
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
        local_rank=args.local_rank,
        dataloader_num_workers=args.dataloader_num_workers,
        disable_tqdm=args.disable_tqdm,
        remove_unused_columns=args.remove_unused_columns,
        deepspeed=args.deepspeed,
        optim=args.optim,
        report_to=args.report_to,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    # 初始化损失函数
    loss_func = TargetLMLoss(ignore_index=-100)

    data_collator = SFTDataCollator(tokenizer, args.max_seq_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
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
