import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from model import GPTConfig, MLP
from einops import rearrange





def kronecker_decompose(A , m: int, n: int, *, k: int = 1, niter: int = 10):
    m2, n2 = A.shape[-2] // m, A.shape[-1] // n
    assert A.shape[-2:] == (m * m2, n * n2), "Dimensions do not match"

    # Reshape and permute A, then perform SVD
    A = rearrange(A, "... (m m2) (n n2) -> ... (m n) (m2 n2)", m=m, m2=m2, n=n, n2=n2)
    u, s, v = torch.svd_lowrank(A, q=k, niter=niter)

    # Unflatten the factors
    u = rearrange(u, "... (m n) k -> ... k m n", m=m, n=n, k=k)
    v = rearrange(v, "... (m2 n2) k -> ... k m2 n2", m2=m2, n2=n2, k=k)

    scale = s[..., None, None].sqrt()
    return u * scale, v * scale


print(f"gpu burn")
device = torch.device("cuda:2")
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

# copies original weights of block_h
def origin_init_block_h(checkpoint, block_num : int):
    mlp_c_fc =  f"_orig_mod.transformer.h.{block_num}.mlp.c_fc.weight"
    mlp_c_proj = f"_orig_mod.transformer.h.{block_num}.mlp.c_proj.weight"
    return checkpoint[mlp_c_fc] ,checkpoint[mlp_c_proj]

# copies Van Loan Decomposition of block_h
def VL_init_block_h(VL_init_all, block_num : int):
    return  {i[20:]:VL_init_all[i] for i in VL_init_all if f"h.{block_num}.mlp" in i}




w_cfc, w_cproj = origin_init_block_h(origin_init_all, 0)

w_cfc1, w_cfc2 = kronecker_decompose(w_cfc, 1536,32)
w_cproj1, w_cproj2 = kronecker_decompose(w_cproj, 32, 1536)

print(f"shape check")
print(w_cfc1.shape, w_cfc2.shape)     
print(w_cproj1.shape, w_cproj2.shape) 

w_cfc1, w_cfc2     =   w_cfc1.squeeze(0), w_cfc2.squeeze(0)
w_cproj1, w_cproj2 = w_cproj1.squeeze(0), w_cproj2.squeeze(0)

print(f"shape check")
print(w_cfc1.shape, w_cfc2.shape)     
print(w_cproj1.shape, w_cproj2.shape) 

check1 = VL_init_block_h(VL_init_all, 0)

check1["c_fc_1"]   = w_cfc1
check1["c_fc_2"]   = w_cfc2

check1["c_proj_1"] = w_cproj1
check1["c_proj_2"] = w_cproj2



# Configuring Krony
print(f"kronyGPT conf")
conf = GPTConfig(**args)
model = KronyMLP(conf)

pre_trained_MLP = MLP(conf)


print(f"kronyGPT state dict loading from VL_init")
model.load_state_dict(check1)
# Data Stuff

print(f"Data loading conf")
train_data = np.memmap('data/shakespeare_char/train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('data/shakespeare_char/val.bin', dtype=np.uint16, mode='r')


block_size = 256
batch_size = 64

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)

    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y


X, Y = get_batch('train')
