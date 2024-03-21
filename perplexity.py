# this script computes the perplexity of a gpt2-like model.
# the model has to be in a ./hf/model

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import torch
from tqdm import tqdm
from model import *
from model_origin import *

import sys
import numpy as np  

device = "cuda"

model_dir = sys.argv[1]  # model directory, usually a number. Model usually stored in ./hf/number
data      = sys.argv[2]  # wiki103 wiki2 lambada 1 2 3

model  = GPT2LMHeadModel.from_pretrained(f"./hf/{model_dir}").to(device)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

wiki = ['wikitext-103-v1', 'wikitext-2-v1']
def test_this(dataset, model):
    if dataset == "wiki103":
        test = load_dataset("wikitext", wiki[0], split="test")
    elif dataset == "wiki2":
        test = load_dataset("wikitext", wiki[1], split="test")
    elif dataset == "lambada":
        test = load_dataset("lambada", split="test")

    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = model.config.n_positions
    stride = 1024
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

if data == "wiki103":
    ppl = test_this(data,   model)
    print(f"ppl on {data} is {ppl}")
elif data == "wiki2":
    ppl = test_this(data,   model)
    print(f"ppl on {data} is {ppl}")
elif data == "lambada":
    ppl = test_this(data,   model)
    print(f"ppl on {data} is {ppl}")
else:
    data = "wiki103"
    ppl = test_this("wiki103",   model)
    print(f"ppl on {data} is {ppl}")
    data = "wiki2"
    ppl = test_this(data,   model)
    print(f"ppl on {data} is {ppl}")
    data = "lambada"
    ppl = test_this(data,   model)
    print(f"ppl on {data} is {ppl}")
        


