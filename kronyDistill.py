import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from model import GPTConfig

device = torch.device("cuda:2")


'''
train_data = np.memmap('data/shakespeare_char/train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('data/shakespeare_char/val.bin', dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y


X, Y = get_batch('train')
'''
print(f"gpu burn")
device = "cuda:3"
x = torch.randn(500,500, device = device)
_ = x@x
print(f"DONE")


print(f"loading ckpt-KP and ckpt")
checkpoint_VL = torch.load('out-shakespeare-char/ckpt-KP.pt')
checkpoint_origin =  torch.load('out-shakespeare-char/ckpt.pt')
print(f"DONE")

VL_init_all = checkpoint_VL["model"]
origin_init_all = checkpoint_origin["model"]



def print_check(checkpoint):  
    for i in checkpoint: print(i)


def distill_block_number(i : int) -> None:
    pass


args = checkpoint_origin["model_args"]

class KronyMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc_1    = nn.Parameter(torch.zeros(1536,32))
        self.c_fc_2    = nn.Parameter(torch.zeros(1, 12))
        self.gelu    = nn.GELU()
        self.c_proj_1  = nn.Parameter(torch.zeros(32,1536))
        self.c_proj_2  = nn.Parameter(torch.zeros(12,1))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        kr1 = torch.kron(self.c_fc_1, self.c_fc_2)
        kr2 = torch.kron(self.c_proj_1, self.c_proj_2)
        x = x@kr1
        x = self.gelu(x)
        x = x@kr2
        x = self.dropout(x)
        return x

conf = GPTConfig(**args)
model = KronyMLP(conf)


# original weight matrices of block h
def origin_init_block_h(checkpoint, block_num : int):
    mlp_c_fc =  f"_orig_mod.transformer.h.{block_num}.mlp.c_fc.weight"
    mlp_c_proj = f"_orig_mod.transformer.h.{block_num}.mlp.c_proj.weight"
    return checkpoint[mlp_c_fc] ,checkpoint[mlp_c_proj]

c_fc_0, c_proj_0 = origin_init_block_h(origin_init_all, 0)

# the original decomposition, weirdly 0. 
def VL_init_block_h(VL_init_all, block_num : int):
    return  {i[22:]:VL_init_all[i] for i in VL_init_all if f"h.{block_num}.mlp" in i}


        
