#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2ForSequenceClassification
from transformers.models.deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer


def get_args():
    """
    python3 4.test_model.py --pretrained_model_name_or_path /data/tianxing/PycharmProjects/Transformers/trained_models/reward_model_gpt2_stack

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default=(project_path / "trained_models/reward_model_gpt2_stack").as_posix(),
        type=str
    )
    parser.add_argument("--question", default="I know the question has been asked thousands of times, but I'll ask it again: is there a way (even patchy) to write/read a dumb text file with Javascript or Protoype ? This is only for debug purposes, and is not designed for production. The thing is I need it to work with (at least) both Firefox and IE (preferably under Windows). Thanks in advance !", type=str)
    parser.add_argument(
        "--response_j",
        default="**It *is* possible to read/write to a local file via JavaScript**: take a look at [TiddlyWIki](http://www.tiddlywiki.com/). *(Caveat: only works for local documents.)* I have actually written a [Single Page Application](http://softwareas.com/towards-a-single-page-application-framework) (SPA) using [twFile](http://jquery.tiddlywiki.org/twFile.html), a part of the TiddlyWiki codebase: 1. Works in different browsers: (IE, Firefox, Chrome) 2. This code is a little old now. TiddlyWiki abandoned the jQuery plugin design a while ago. (Look at the [current TiddlyWiki filesystem.js](http://dev.tiddlywiki.org/browser/Trunk/core/js/FileSystem.js) for more a more recent implementation. It's not isolated for you like the twFile plug-in, though). 3. Although written as a jQuery plug-in, I've studied the code and it is almost completely decoupled from jQuery. **Update:** I have uploaded a [proof-of-concept](http://coolcases.com/jeopardy/) that accesses a local file via JavaScript. * Modifying this application to write to a file is trivial. * I have not tried to get this to work as a file served from a web server, but it should be possible since there are [server-side implementations of TiddlyWiki](http://tiddlywiki.org/wiki/Can_I_use_TiddlyWiki_as_a_multi-user/collaborative/server_based_wiki%3F)<>. **Update:** So it looks like the server side implementations of TiddlyWiki use a server \"adapter\" to modify a file stored on the server, similar to [Peter's description](https://stackoverflow.com/questions/3195720/write-a-file-with-prototype-or-plain-javascript/3195752#3195752). The pure JavaScript method will probably not work if the page is served from a web server due to cross-domain security limitations.",
        type=str
    )
    parser.add_argument(
        "--response_k",
        default="Javascript in browsers doesn't allow you to write local files, for **security reasons**. This **may change with time**, but as for now you have to **deal with it**.",
        type=str
    )
    parser.add_argument('--max_length', default=512, type=int)

    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model_name_or_path,
        num_labels=1,
    )
    model.eval()

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    text_j = "Question: {}\n\nAnswer: {}".format(args.question, args.response_j)
    text_k = "Question: {}\n\nAnswer: {}".format(args.question, args.response_k)

    text_encoded = tokenizer.__call__([text_j, text_k],
                                      padding="longest",
                                      max_length=args.max_length,
                                      truncation=True
                                      )

    input_ids = text_encoded["input_ids"]
    attention_mask = text_encoded["attention_mask"]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)

    outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask)
    pooled_logits = outputs[0]
    pooled_logits = pooled_logits.cpu().detach()
    score = nn.functional.sigmoid(pooled_logits)

    print(score.shape)
    print(score)

    return


if __name__ == '__main__':
    main()
