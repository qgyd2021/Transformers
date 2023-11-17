#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/datasets/qgyd2021/sentence_pair

https://huggingface.co/shibing624/text2vec-base-chinese

reference:
https://huggingface.co/blog/how-to-train-sentence-transformers
https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/95_Training_Sentence_Transformers.ipynb
https://www.sbert.net/docs/training/overview.html
https://huggingface.co/blog/1b-sentence-embeddings
https://github.com/shibing624/text2vec


train on kaggle notebook with single GPU.

from shutil import copyfile
copyfile(src = "../input/tempdataset/step_2_train_model.py", dst = "../working/step_2_train_model.py")

import step_2_train_model
step_2_train_model.train_on_kaggle_notebook()

SentenceTransformers seems not support multi GPU training:
https://github.com/UKPLab/sentence-transformers/pull/2338
in this pull we can find that she implements multi-GPU training through PyTorch Lighting.

"""
import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import platform
import re

if platform.system() == "Windows":
    from project_settings import project_path
else:
    project_path = os.path.abspath("./")
    project_path = Path(project_path)

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

import huggingface_hub
from sentence_transformers import SentenceTransformer, models
from sentence_transformers import InputExample
from sentence_transformers import losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import torch
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="hfl/chinese-macbert-base",
        type=str
    )

    parser.add_argument(
        "--train_subset",
        default="data_dir/train.jsonl",
        type=str
    )
    parser.add_argument(
        "--valid_subset",
        default="data_dir/valid.jsonl",
        type=str
    )

    parser.add_argument("--num_epochs", default=40, type=int)
    parser.add_argument(
        "--hf_token",
        default=None,
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    huggingface_hub.login(token=args.hf_token)

    # model
    word_embedding_model = models.Transformer(model_name_or_path=args.pretrained_model_name_or_path)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model],
        use_auth_token=args.hf_token,
    )

    # dataset
    train_examples = []
    with open(args.train_subset, "r", encoding="utf-8") as f:
        for row in f:
            row = json.loads(row)
            train_examples.append(InputExample(texts=[row["text1"], row["text2"]], label=int(row["label"])))
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    valid_examples = []
    with open(args.valid_subset, "r", encoding="utf-8") as f:
        for row in f:
            row = json.loads(row)
            valid_examples.append(InputExample(texts=[row["text1"], row["text2"]], label=int(row["label"])))

    valid_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(valid_examples, batch_size=16)

    dataset_info = f"""
    train dataset: {len(train_examples)}
    """
    dataset_info = re.sub(r"[\u0020]{4,}", "", dataset_info)
    print(dataset_info)

    # loss
    train_loss = losses.ContrastiveLoss(
        model=model,
        distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE
    )

    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1)

    # train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=valid_evaluator,
        epochs=args.num_epochs,
        evaluation_steps=int(len(train_dataloader) * 0.1),
        scheduler="WarmupLinear",
        warmup_steps=warmup_steps,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={"lr": 2e-5},
        output_path="output_path",
        save_best_model=True,
        max_grad_norm=1.,
        use_amp=False,
        checkpoint_save_steps=int(len(train_dataloader) * 0.1),
        checkpoint_save_total_limit=2,
    )
    return


if __name__ == '__main__':
    main()
