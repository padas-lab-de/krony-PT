from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import torch
from tqdm import tqdm
from model import *
from model_origin import *

import numpy as np  

device = "cuda"
model  = GPT2LMHeadModel.from_pretrained("./hf/13").to(device)
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

results = {}
for elt in ["wiki103", "wiki2", "lambada"]:
    print(f"Computing perplexity for {elt}")
    ppl = test_this(elt,   model)
    results[elt] = ppl
    print(ppl)
    
print(results)

