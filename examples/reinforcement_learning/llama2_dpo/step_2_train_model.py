#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
reference
https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py

https://huggingface.co/blog/dpo-trl
https://huggingface.co/spaces/trl-lib/stack-llama

dataset:
https://huggingface.co/datasets/lvwerra/stack-exchange-paired
"""
from dataclasses import dataclass, field
import os
from typing import Dict, Optional

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from trl import DPOTrainer


@dataclass
class ScriptArguments:
    # dataset
    dataset_path: str = field(default="lvwerra/stack-exchange-paired")
    dataset_name: str = field(default=None)
    dataset_data_dir: str = field(default="data/rl")
    dataset_split: str = field(default="train")
    dataset_cache_dir: str = field(default=(project_path / "hub_datasets").as_posix())
    # train_subset: Optional[int] = field(default=-1)
    # eval_subset: Optional[int] = field(default=10000)

    # cache
    cache_dir: str = field(default="cache_dir")

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})

    model_name_or_path: Optional[str] = field(
        default="qgyd2021/sft_llama2_stack_exchange",
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(default="tensorboard")

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
                    "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


def get_args():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args


def get_stack_exchange_paired(path: str,
                              name: str,
                              split: str,
                              cache_dir: str,
                              data_dir: str,
                              max_length: int,
                              sanity_check: bool = True,
                              num_proc: int = None
                              ):
    dataset = load_dataset(
        path=path,
        name=name,
        split=split,
        cache_dir=cache_dir,
        data_dir=data_dir,
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": ["Question: " + question + "\n\nAnswer: " for question in samples["question"]],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

    dataset = dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_length and len(x["prompt"]) + len(x["rejected"]) <= max_length
    )

    return dataset


def main():
    args = get_args()

    # dataset
    train_dataset = get_stack_exchange_paired(
        path=args.dataset_path,
        name=args.dataset_name,
        split="train",
        data_dir="data/rl",
        cache_dir=args.dataset_cache_dir,
        max_length=args.max_length,
        sanity_check=args.sanity_check,
        num_proc=os.cpu_count() // 2,
    )

    valid_dataset = get_stack_exchange_paired(
        path=args.dataset_path,
        name=args.dataset_name,
        split="train",
        data_dir="data/evaluation",
        cache_dir=args.dataset_cache_dir,
        max_length=args.max_length,
        sanity_check=args.sanity_check,
        num_proc=os.cpu_count() // 2,
    )

    # model, model reference
    # AutoPeftModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    model.config.use_cache = False

    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # AutoPeftModelForCausalLM
    model_ref = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == "llama" else True,
        padding_side="left"
    )

    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        report_to=args.report_to,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optimizer_type,
        fp16=True,
        remove_unused_columns=False,
        run_name="dpo_llama2",
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
    )

    dpo_trainer.train()
    dpo_trainer.save_model(args.output_dir)

    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
    return


if __name__ == '__main__':
    main()
