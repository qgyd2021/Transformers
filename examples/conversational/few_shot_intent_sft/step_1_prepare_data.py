#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json

from datasets import load_dataset
from datasets.download.download_manager import DownloadMode
from tqdm import tqdm

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="qgyd2021/few_shot_intent_sft", type=str)
    parser.add_argument("--dataset_split", default=None, type=str)
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )

    parser.add_argument("--num_epochs", default=1, type=int)

    parser.add_argument("--train_subset", default="train.jsonl", type=str)
    parser.add_argument("--valid_subset", default="valid.jsonl", type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    name_list = [
        # "a_intent_prompt",
        "amazon_massive_intent_en_us_prompt",
        "amazon_massive_intent_zh_cn_prompt",
        "atis_intents_prompt",
        "banking77_prompt",
        "bi_text11_prompt",
        "bi_text27_prompt",
        # "book6_prompt",
        "carer_prompt",
        "chatbots_prompt",
        "chinese_news_title_prompt",
        "cmid_4class_prompt",
        "cmid_36class_prompt",
        "coig_cqia_prompt",
        "conv_intent_prompt",
        "crosswoz_prompt",
        "dmslots_prompt",
        "dnd_style_intents_prompt",
        "emo2019_prompt",
        "finance21_prompt",
        "ide_intent_prompt",
        "intent_classification_prompt",
        "jarvis_intent_prompt",
        "mobile_assistant_prompt",
        "mtop_intent_prompt",
        "out_of_scope_prompt",
        "ri_sawoz_domain_prompt",
        "ri_sawoz_general_prompt",
        "small_talk_prompt",
        "smp2017_task1_prompt",
        "smp2019_task1_domain_prompt",
        "smp2019_task1_intent_prompt",
        # "snips_built_in_intents_prompt",
        "star_wars_prompt",
        "suicide_intent_prompt",
        "snips_built_in_intents_prompt",
        "telemarketing_intent_cn_prompt",
        "telemarketing_intent_en_prompt",
        "vira_intents_prompt",
    ]

    name_list = [
        "telemarketing_intent_cn_prompt",

    ]

    with open(args.train_subset, "w", encoding="utf-8") as f:
        for _ in range(args.num_epochs):
            for name in name_list:
                print(name)
                dataset = load_dataset(
                    path=args.dataset_path,
                    name=name,
                    split="train",
                    cache_dir=args.dataset_cache_dir,
                    download_mode=DownloadMode.FORCE_REDOWNLOAD,
                    ignore_verifications=True
                )
                for sample in tqdm(dataset):
                    row = json.dumps(sample, ensure_ascii=False)
                    f.write("{}\n".format(row))

    with open(args.valid_subset, "w", encoding="utf-8") as f:
        for _ in range(args.num_epochs):
            for name in name_list:
                print(name)
                dataset = load_dataset(
                    path=args.dataset_path,
                    name=name,
                    split="test",
                    cache_dir=args.dataset_cache_dir,
                    download_mode=DownloadMode.FORCE_REDOWNLOAD,
                    ignore_verifications=True
                )
                for sample in tqdm(dataset):
                    row = json.dumps(sample, ensure_ascii=False)
                    f.write("{}\n".format(row))

    return


if __name__ == '__main__':
    main()
