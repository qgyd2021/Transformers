#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://github.com/thunlp/OpenDelta/blob/main/examples/tutorial/0_interactive.py
"""
import argparse
import os

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from opendelta import LoraModel
from transformers import BertForMaskedLM


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", default="bert-base-cased", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model = BertForMaskedLM.from_pretrained(args.pretrained_model_name_or_path)

    # 执行这行代码时会启动一个服务, 你可以访问 http://127.0.0.1:8888/
    # 点击你想要执行 LoRA 低秩化的 Module, 它被黄色细线包围即被选中.
    # 之后这些 Module 会被 LoRA 修改.
    delta_model = LoraModel(backbone_model=model, interactive_modify=True)

    delta_model.freeze_module(exclude=["deltas", "layernorm_embedding"], set_state_dict=True)
    delta_model.log()

    return


if __name__ == '__main__':
    main()
