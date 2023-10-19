#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://github.com/thunlp/OpenDelta/blob/main/examples/tutorial/0_basic.py
"""
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from bigmodelvis import Visualization
from opendelta import LoraModel
from transformers import AutoModelForSequenceClassification
from transformers.models.bart.modeling_bart import BartForSequenceClassification


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", default="facebook/bart-base", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model: BartForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model_name_or_path)

    print("before modify")
    Visualization(model).structure_graph()

    delta_model = LoraModel(backbone_model=model, modified_modules=['fc2'])
    print("after modify")
    delta_model.log()

    delta_model.freeze_module(exclude=["deltas", "layernorm_embedding"], set_state_dict=True)
    print("after freeze")
    delta_model.log()

    return


if __name__ == '__main__':
    main()
