"""
In this script, we evaluate VL init with 2 factors, and GPT2.
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
    config_args2 = dict(
        n_layer=12, 
        n_head=12, 
        n_embd=768,
        vocab_size = 50257,
        block_size = 1024,
        bias = True,
        factors = 2
    )

    config_args1 = dict(
        n_layer=12, 
        n_head=12, 
        n_embd=768,
        vocab_size = 50257,
        block_size = 1024,
        bias = True,
    )

    block_size = config_args1["block_size"]
    device = "cuda"
    device_type = "cuda"
    eval_iters = 50



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
conf = GPTConfig(**config_args1)
normy_gpt = GPT(conf)

print("Loading checkpoint for GPT2")
checkpoint = torch.load("out/GPT2.pt")
normy_gpt.load_state_dict(checkpoint)

ctx =  torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

# Normy Loss.
normy_gpt.to(device)
print(">> Computing the loss")
#print(f"Loss for NormyGPT is {estimate_loss(normy_gpt)}")

# Case 2:  Kronecker & VL init with 2 factors.
print("Loading KronyGPT with VL init and 2 factors")
sd = torch.load("out/GPT2_VL_2_factors.pt", map_location=device)

krony_conf = KronyGPTConfig(**config_args2)
krony_gpt = KronyGPT(krony_conf)
krony_gpt.load_state_dict(sd)



krony_gpt.to(device)
sd = krony_gpt.state_dict()

print(">> Computing the loss, should be the same")
#print(f"Loss for KronyGPT with VL init is {estimate_loss(krony_gpt)}")

## Losses:
# loss for KronyGPT >> 3.10 and 3.10
# loss for VL init >> 4.23 and 4.22

f = 2 # number of factor

for i in range(12):
    print("\n\n step <<<<<<<<<<<<<<<<<<<<<<<<<<<   \n\n")

    c_fc_key = f"transformer.h.{i}.mlp.c_fc.weight"
    c_proj_key = f"transformer.h.{i}.mlp.c_proj.weight"

    fcW =   checkpoint[c_fc_key].to(device)
    projW = checkpoint[c_proj_key].to(device)   

    fc = torch.zeros(3072, 768).to(device)
    proj = torch.zeros(768, 3072).to(device)

    for f in range(2):
        fc1   = sd[f"transformer.h.{i}.mlp.c_fc_{f}_{0}"].to(device)
        fc2   = sd[f"transformer.h.{i}.mlp.c_fc_{f}_{1}"].to(device)
        fc += torch.kron(fc1, fc2)

        proj1   = sd[f"transformer.h.{i}.mlp.c_proj_{f}_{0}"].to(device)
        proj2   = sd[f"transformer.h.{i}.mlp.c_proj_{f}_{1}"].to(device)
        proj += torch.kron(proj1, proj2)

    print(torch.max(torch.abs(fcW- fc)))
    print(torch.max(torch.abs(projW- proj)))

'''
s_cfc =  [
    torch.kron(getattr(self, f"c_fc_{f}_0"), getattr(self, f"c_fc_{f}_1"))
    for f in range(2)
]

s_cproj =  [
    torch.kron(getattr(self, f"c_proj_{f}_0"), getattr(self, f"c_proj_{f}_1"))
    for f in range(2)
]
'''