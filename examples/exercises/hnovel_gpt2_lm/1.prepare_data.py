#!/usr/bin/python3
# -*- coding: utf-8 -*-
from datasets import load_dataset


dataset = load_dataset(
    path="qgyd2021/HNovel",
    name="ltxsba",
    streaming=True,
)
for sample in dataset["train"]:
    print(sample)


if __name__ == '__main__':
    pass
