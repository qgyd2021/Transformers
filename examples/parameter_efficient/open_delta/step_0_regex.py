#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://github.com/thunlp/OpenDelta/blob/main/examples/tutorial/0_regex.py
"""
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from bigmodelvis import Visualization
from opendelta import LoraModel
from transformers import AutoModelForSequenceClassification


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", default="roberta-base", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model_name_or_path)

    Visualization(model).structure_graph()

    delta_model = LoraModel(
        backbone_model=model,
        modified_modules=[r"[r](\d)+\.output\.dense", "attention.output.dense"]
    )

    # delta_model = LoraModel(
    #     backbone_model=model,
    #     modified_modules=[r"[r][0-5]\.output\.dense"]
    # )

    delta_model.log()

    return


if __name__ == '__main__':
    main()
