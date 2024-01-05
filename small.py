"""
Evaluation of KronyGPT with 3 initializations:
1. Van Loan.
2. Random.
3. MaxPooling with tricky trick.
"""

import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from transformers import GPT2LMHeadModel

from model_origin import *
from model import *

# TRY to keep this script self-contained > testing different setting for init.

if True:
    # put some vars here.
    config_args = dict(
        n_layer=12, 
        n_head=12, 
        n_embd=768,
        vocab_size = 50257,
        block_size = 1024,
        bias = True,
    )

    block_size = config_args["block_size"]
    device = "cuda"
    device_type = "cuda"
    eval_iters = 200



if True: # data loader here AND estimate loss.
    path = 'data/openwebtext/'
    train_data = np.memmap(f'{path}train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap(f'{path}val.bin', dtype=np.uint16, mode='r')

    batch_size = 12
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out


# Case 1: Normy GPT
conf = GPTConfig(**config_args)
normy_gpt = GPT(conf)

# loading the checkpoints
checkpoint = torch.load("out/GPT2.pt")
normy_gpt.load_state_dict(checkpoint)

ctx =  torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

# Normy Loss.
normy_gpt.to(device)
print(f"Loss for NormyGPT is {estimate_loss(normy_gpt)}")

# Case 2:  Kronecker & VL init.
print("KronyGPT with VL init:")
sd = torch.load("out/GPT2_VL11.pt")
print(">> Loading DONE")

# this would would be the same for all.
krony_conf = KronyGPTConfig(**config_args)

krony_gpt = KronyGPT(krony_conf)
krony_gpt.load_state_dict(sd)
krony_gpt.to(device)
print(f"Loss for KronyGPT with VL init is {estimate_loss(krony_gpt)}")

# GPT2 with Kronecker & Random init

# write custom code for inits
# write custom code for lr


# GPT2 with Kronecker & simple 1/2 prunning init
print("Loading KronyGPT with prune 1/2 init:")
sd_prune = torch.load("out/GPT2_prune_init.pt")
print(">> Loading DONE")



krony_gpt_prune = KronyGPT(krony_conf)
krony_gpt_prune.load_state_dict(sd_prune)
krony_gpt_prune.to(device)
print(f"Loss for KronyGPT with prune 1/2 init {estimate_loss(krony_gpt_prune)}")



