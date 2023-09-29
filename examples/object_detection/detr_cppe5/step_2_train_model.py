#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
reference:
https://huggingface.co/docs/transformers/tasks/object_detection

pip install -q datasets transformers evaluate timm albumentations

"""
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Dict, List

# from project_settings import project_path
# project_path = os.path.abspath(os.path.dirname(__file__))
project_path = os.path.abspath("./")
project_path = Path(project_path)

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

import albumentations
from datasets import load_dataset
import huggingface_hub
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import HfArgumentParser
from transformers.models.auto.processing_auto import AutoImageProcessor
from transformers.models.auto.modeling_auto import AutoModelForObjectDetection
from transformers import TrainingArguments
from transformers import Trainer


@dataclass
class ScriptArguments:
    # dataset
    dataset_path: str = field(default="qgyd2021/cppe-5")
    dataset_name: str = field(default=None)
    dataset_cache_dir: str = field(default=(project_path / "hub_datasets").as_posix())
    # dataset_cache_dir: str = field(default="hub_datasets")

    # model
    pretrained_model_name_or_path: str = field(default="facebook/detr-resnet-50")

    # training_args
    output_dir: str = field(default="output_dir")
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=4)
    num_train_epochs: float = field(default=30)
    fp16: bool = field(default=True)
    save_steps: int = field(default=200)
    logging_steps: int = field(default=50)
    learning_rate: float = field(default=1e-5)
    weight_decay: float = field(default=1e-4)
    save_total_limit: int = field(default=2)
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="tensorboard")
    push_to_hub: bool = field(default=True)
    hub_model_id: str = field(default="detr_cppe5_object_detection")
    hub_strategy: str = field(default="every_save")

    # hf_token
    hf_token: str = field(default="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")


def get_args():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    return args


def show_first_image(example: dict, index_to_label: Dict[int, str]):
    image: Image = example["image"]
    annotations = example["objects"]

    draw = ImageDraw.Draw(image)

    for i in range(len(annotations["id"])):
        box = annotations["bbox"][i - 1]
        class_idx = annotations["category"][i - 1]
        x, y, w, h = tuple(box)
        draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
        draw.text((x, y), index_to_label[class_idx], fill="white")
    return image


def formatted_annotations(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations


def train_model(local_rank, world_size, args):
    os.environ["RANK"] = f"{local_rank}"
    os.environ["LOCAL_RANK"] = f"{local_rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    huggingface_hub.login(token=args.hf_token)

    # dataset
    dataset_dict = load_dataset(
        path=args.dataset_path,
        cache_dir=args.dataset_cache_dir
    )
    train_dataset = dataset_dict["train"]

    remove_idx = [590, 821, 822, 875, 876, 878, 879]
    keep = [i for i in range(len(train_dataset)) if i not in remove_idx]
    train_dataset = train_dataset.select(keep)

    categories = ["Coverall", "Face_Shield", "Gloves", "Goggles", "Mask"]
    index_to_label = {index: x for index, x in enumerate(categories, start=0)}
    label_to_index = {v: k for k, v in index_to_label.items()}

    # first_example = train_dataset[0]
    # image: Image = show_first_image(example=first_example, index_to_label=index_to_label)
    # image.show()

    image_processor = AutoImageProcessor.from_pretrained(args.pretrained_model_name_or_path)

    transform = albumentations.Compose(
        [
            albumentations.Resize(480, 480),
            albumentations.HorizontalFlip(p=1.0),
            albumentations.RandomBrightnessContrast(p=1.0),
        ],
        bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
    )

    # transforming a batch
    def transform_aug_annotation(examples):
        image_ids = examples["image_id"]
        images, bboxes, area, categories = [], [], [], []
        for image, objects in zip(examples["image"], examples["objects"]):
            image = np.array(image.convert("RGB"))[:, :, ::-1]
            out = transform.__call__(image=image, bboxes=objects["bbox"], category=objects["category"])

            area.append(objects["area"])
            images.append(out["image"])
            bboxes.append(out["bboxes"])
            categories.append(out["category"])

        targets = [
            {"image_id": id_, "annotations": formatted_annotations(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
        ]

        return image_processor.__call__(images=images, annotations=targets, return_tensors="pt")

    train_dataset = train_dataset.with_transform(transform_aug_annotation)

    def collate_fn(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        encoding = image_processor.pad(pixel_values, return_tensors="pt")
        labels = [item["labels"] for item in batch]
        batch = {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels
        }
        return batch

    model = AutoModelForObjectDetection.from_pretrained(
        args.pretrained_model_name_or_path,
        id2label=index_to_label,
        label2id=label_to_index,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        fp16=args.fp16,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=args.remove_unused_columns,
        report_to=args.report_to,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy=args.hub_strategy,
        local_rank=local_rank,
        ddp_backend="nccl",
        # fsdp="auto_wrap",
    )
    print(training_args)

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

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        tokenizer=image_processor,
    )
    trainer.train()
    trainer.push_to_hub()
    return


def single_gpu_train():
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
    single_gpu_train()
