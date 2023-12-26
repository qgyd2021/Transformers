#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json

from datasets import load_dataset
from tqdm import tqdm

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="qgyd2021/few_shot_ner_sft", type=str)
    parser.add_argument("--dataset_split", default=None, type=str)
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )
    parser.add_argument("--train_subset", default="train.jsonl", type=str)
    parser.add_argument("--valid_subset", default="valid.jsonl", type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    name_list = [
        "acronym_identification_prompt",
        "bank_prompt",
        "bc4chemd_ner_prompt",
        "bc2gm_prompt",
        "ccfbdci_prompt",
        "ccks2019_task1_prompt",
        "cluener2020_prompt",
        "cmeee_prompt",
        "conll2003_prompt",
        "conll2012_ontonotesv5_chinese_v4_prompt",
        "conll2012_ontonotesv5_english_v4_prompt",
        "conll2012_ontonotesv5_english_v12_prompt",
        "dlner_prompt",
        "ecommerce_prompt",
        "episet4ner_v2_prompt",
        "few_nerd_inter_prompt",
        "few_nerd_inter_fine_prompt",
        "few_nerd_intra_prompt",
        "few_nerd_intra_fine_prompt",
        "few_nerd_supervised_prompt",
        "few_nerd_supervised_fine_prompt",
        "finance_sina_prompt",
        "limit_prompt",
        "msra_prompt",
        "ncbi_disease_prompt",
        "nlpcc2018_task4_prompt",
        "people_daily_prompt",
        "pet_prompt",
        # "plod_prompt",
        "resume_prompt",
        "sd_nlp_non_tokenized_prompt",
        "wiesp2022_ner_prompt",
        "weibo_prompt",
        "wnut_17_prompt",
        "xtreme_en_prompt",
        "youku_prompt",

    ]

    with open(args.train_subset, "w", encoding="utf-8") as f:
        for name in name_list:
            dataset = load_dataset(
                path=args.dataset_path,
                name=name,
                split="train",
                cache_dir=args.dataset_cache_dir
            )

            for sample in tqdm(dataset):
                row = json.dumps(sample, ensure_ascii=False)
                f.write("{}\n".format(row))

    with open(args.valid_subset, "w", encoding="utf-8") as f:
        for name in name_list:
            dataset = load_dataset(
                path=args.dataset_path,
                name=name,
                split="test",
                cache_dir=args.dataset_cache_dir
            )

            for sample in tqdm(dataset):
                row = json.dumps(sample, ensure_ascii=False)
                f.write("{}\n".format(row))

    return


if __name__ == '__main__':
    main()
