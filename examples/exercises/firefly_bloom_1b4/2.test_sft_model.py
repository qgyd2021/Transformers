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

    参考链接:
    https://huggingface.co/YeungNLP/firefly-bloom-1b4

    Example:
        将下面句子翻译成现代文：\n石中央又生一树，高百余尺，条干偃阴为五色，翠叶如盘，花径尺余，色深碧，蕊深红，异香成烟，著物霏霏。

        实体识别: 1949年10月1日，人们在北京天安门广场参加开国大典。

        把这句话翻译成英文: 1949年10月1日，人们在北京天安门广场参加开国大典。

        晚上睡不着该怎么办. 请给点详细的介绍.

        将下面的句子翻译成文言文：结婚率下降, 离婚率暴增, 生育率下降, 人民焦虑迷茫, 到底是谁的错.

        对联：厌烟沿檐烟燕眼. (污雾舞坞寤梧芜).

        写一首咏雪的古诗, 标题为 "沁园春, 雪".

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
