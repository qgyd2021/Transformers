#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
from typing import List

from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self):
        self.samples: List[dict] = list()

    def read(self, filename: str):
        samples = list()
        with open(filename, "r", encoding="utf-8") as f:
            for row in f:
                row = str(row).strip()
                row = json.loads(row)
                samples.append(row)
        self.samples = samples
        return self

    def __getitem__(self, index):
        sample = self.samples[index]
        return sample

    def __len__(self):
        return len(self.samples)


class ChatGLM2SFTDataset(SFTDataset):
    def __init__(self, tokenizer, max_seq_length: int):
        super(ChatGLM2SFTDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.input_format = '[Round {}]\n\n问：{}\n\n答：'
        self.target_format = "{}"

    def __getitem__(self, index):
        sample = self.samples[index]

        conversation = sample["conversation"]

        utterances = list()
        for i, x in enumerate(conversation):
            human = self.input_format.format(i+1, x["human"])
            assistant = self.target_format.format(x["assistant"])
            utterances += ([human, assistant])
        utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids

        input_ids = list()
        target_mask = list()
        for i, utterances_id in enumerate(utterances_ids):
            input_ids += utterances_id
            if i % 2 == 0:
                target_mask += [0] * (len(utterances_id))
            else:
                input_ids += [self.tokenizer.eos_token_id]
                target_mask += [1] * (len(utterances_id) + 1)

        assert len(input_ids) == len(target_mask)

        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)

        assert len(input_ids) == len(target_mask) == len(attention_mask)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask
        }
        return inputs


if __name__ == '__main__':
    pass
