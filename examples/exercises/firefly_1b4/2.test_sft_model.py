#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

import torch
from transformers import BloomTokenizerFast, BloomForCausalLM

from project_settings import project_path


def get_args():
    """
    python3 2.test_sft_model.py --trained_model_path /data/tianxing/PycharmProjects/Transformers/trained_models/bloom-396m-sft
    python3 2.test_sft_model.py --trained_model_path /data/tianxing/PycharmProjects/Transformers/trained_models/bloom-1b4-sft

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trained_model_path',
        # default='YeungNLP/bloom-1b4-zh',
        default=(project_path / "trained_models/bloom-1b4-sft").as_posix(),
        type=str,
    )
    parser.add_argument('--device', default='auto', type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # pretrained model
    tokenizer = BloomTokenizerFast.from_pretrained(args.trained_model_path)
    model = BloomForCausalLM.from_pretrained(args.trained_model_path)

    model.eval()
    model = model.to(device)
    text = input('User：')
    while True:
        text = '<s>{}</s></s>'.format(text)
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        outputs = model.generate(input_ids, max_new_tokens=200, do_sample=True, top_p=0.85, temperature=0.35,
                                 repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
        rets = tokenizer.batch_decode(outputs)
        output = rets[0].strip().replace(text, "").replace('</s>', "")
        print("LLM：{}".format(output))
        text = input('User：')


if __name__ == '__main__':
    main()
