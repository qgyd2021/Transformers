#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", default="lh .txt", type=str)
    parser.add_argument("--train_subset", default="train.txt", type=str)
    parser.add_argument("--valid_subset", default="valid.txt", type=str)

    parser.add_argument("--min_chars", default=512, type=int)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    with open(args.corpus_file, "r", encoding="utf-8") as fin, \
        open(args.train_subset, "w", encoding="utf-8") as ftrain, \
        open(args.valid_subset, "w", encoding="utf-8") as fvalid:

        row = ""
        for line in fin:
            line = str(line).strip()
            if len(line) == 0:
                continue

            row += line
            if len(row) < args.min_chars:
                continue

            if random.random() < 0.8:
                ftrain.write("{}\n".format(line))
            else:
                fvalid.write("{}\n".format(line))

    return


if __name__ == '__main__':
    main()
