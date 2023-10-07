#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
reference:
https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/sft_llama2.py

"""
from dataclasses import dataclass, field
import os
import platform
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset


@dataclass
class ScriptArguments:

    # dataset
    dataset_path: Optional[str] = field(default="lvwerra/stack-exchange-paired")
    dataset_data_dir: Optional[str] = field(default="data/finetune")
    dataset_split: Optional[str] = field(default="train")
    dataset_streaming: Optional[bool] = field(default=True)

    num_workers: Optional[int] = field(default=None if platform.system() == "Windows" else os.cpu_count() // 2)
    valid_dataset_size: Optional[int] = field(default=10000)
    shuffle_buffer_size: Optional[int] = field(default=20000)

    # ConstantLengthDataset
    seq_length: Optional[int] = field(default=1024)

    # pretrained model
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")

    # lora
    lora_rank: Optional[int] = field(default=8)
    lora_alpha: Optional[float] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)

    # train args
    output_dir: Optional[str] = field(default="output_dir")
    per_device_train_batch_size: Optional[int] = field(default=8)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=1e-4)
    weight_decay: Optional[float] = field(default=0.05)
    max_steps: Optional[int] = field(default=5000)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    warmup_steps: Optional[int] = field(default=500)
    logging_steps: Optional[int] = field(default=100)
    save_steps: Optional[int] = field(default=100)
    save_total_limit: Optional[int] = field(default=2)
    bf16: Optional[bool] = field(default=False)
    fp16: Optional[bool] = field(default=True)
    remove_unused_columns: Optional[bool] = field(default=True)
    optim: Optional[str] = field(default="paged_adamw_32bit")
    group_by_length: Optional[bool] = field(default=False)
    report_to: Optional[str] = field(default="tensorboard")
    gradient_checkpointing: Optional[bool] = field(default=True)

    # trainer
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})


def get_args():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    return text


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        path=args.dataset_path,
        data_dir=args.dataset_data_dir,
        split=args.dataset_split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.dataset_streaming else None,
        streaming=args.dataset_streaming,
    )
    if args.dataset_streaming:
        print("Loading the dataset in streaming mode")
        valid_dataset = dataset.take(args.valid_dataset_size)
        train_dataset = dataset.skip(args.valid_dataset_size)
        train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer_size, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=None)
        train_dataset = dataset["train"]
        valid_dataset = dataset["test"]
        print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(valid_dataset)}")

    chars_per_token = chars_token_ratio(train_dataset, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_dataset,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_dataset,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


def main():
    args = get_args()

    if args.group_by_length and args.packing:
        raise ValueError("Cannot use both packing and group by length")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map={"": Accelerator().local_process_index},
        trust_remote_code=True,
        use_auth_token=True,
    )
    base_model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    train_dataset, eval_dataset = create_datasets(tokenizer, args)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        remove_unused_columns=args.remove_unused_columns,
        optim=args.optim,
        group_by_length=args.group_by_length,
        report_to=args.report_to,
        gradient_checkpointing=args.gradient_checkpointing,
        run_name="sft_llama2",
    )

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=args.packing,
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args,
    )
    trainer.train()
    trainer.save_model(args.output_dir)

    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del base_model
    torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(args.output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    return


if __name__ == '__main__':
    main()
