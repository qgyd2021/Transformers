#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.bert.tokenization_bert import BertTokenizer

from project_settings import project_path


def get_args():
    """
    python3 3.test_model.py --trained_model_path /data/tianxing/PycharmProjects/Transformers/examples/exercises/h_novel_gpt2_lm/file_dir/serialization_dir/final

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trained_model_path',
        default=(project_path / "pretrained_models/gpt2-chinese-cluecorpussmall").as_posix(),
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
    tokenizer = BertTokenizer.from_pretrained(args.trained_model_path)
    model = GPT2LMHeadModel.from_pretrained(args.trained_model_path)

    model.eval()
    model = model.to(device)

    while True:
        text = input('prefix: ')

        if text == "Quit":
            break
        text = '{}'.format(text)
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = input_ids[:, :-1]
        # print(input_ids)
        # print(type(input_ids))
        input_ids = input_ids.to(device)

        outputs = model.generate(input_ids,
                                 max_new_tokens=200,
                                 do_sample=True,
                                 top_p=0.85,
                                 temperature=0.35,
                                 repetition_penalty=1.2,
                                 eos_token_id=tokenizer.sep_token_id,
                                 pad_token_id=tokenizer.pad_token_id
                                 )
        rets = tokenizer.batch_decode(outputs)
        output = rets[0].replace(" ", "").replace("[CLS]", "").replace("[SEP]", "")
        print("{}".format(output))

    return


if __name__ == '__main__':
    main()
