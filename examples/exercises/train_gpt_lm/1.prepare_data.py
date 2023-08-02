#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", default="四书五经.txt", type=str)
    parser.add_argument("--train_subset", default="train.txt", type=str)

    parser.add_argument("--min_chars", default=512, type=int)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # ANSI is not a python build-in encoding type, use ISO-8859-1 instead.
    with open(args.corpus_file, "r", encoding="gbk") as fin, \
        open(args.train_subset, "w", encoding="utf-8") as ftrain:

        row = ""
        for line in fin:
            line = str(line).strip()
            if len(line) == 0:
                continue

            row += line
            if len(row) < args.min_chars:
                continue

            ftrain.write("{}\n".format(line))

    return


if __name__ == '__main__':
    main()
