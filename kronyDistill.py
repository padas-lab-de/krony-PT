import os
import time
import math
import pickle
#from contextlib import nullcontext

import torch
import torch.nn as nn
import numpy as np

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


device = torch.device("cuda:2")
x = torch.randn(500,500, device = device)
_ = x@x
print(f">>> gpu ready")


print(f"loading ckpt-KP and ckpt")
checkpoint_VL = torch.load('out-shakespeare-char/ckpt_VL.pt'  , map_location = device)
checkpoint_origin =  torch.load('out-shakespeare-char/ckpt.pt', map_location = device)
print(f">>> DONE")

origin = checkpoint_origin["model"]
VL  = checkpoint_VL["model"] 


unwanted_prefix = '_orig_mod.'
for k,v in list(origin.items()):
    if k.startswith(unwanted_prefix):
        origin[k[len(unwanted_prefix):]] = origin.pop(k)


origin_params = list(origin.keys())
VL_params = list(VL.keys())



# attack plan: 1. add batch norm layers.  2. pray to the DL gods.
n_embd  = 384
dropout = 0.2
bias = False

def get_from_state_dict(ind : int, origin, VL):
    """
    get the params from checkpoint["model"], with correct keys to be used as model.load_state_dict()
    """
    cfc, cproj = "c_fc", "c_proj"
    pref_cfc  = f"transformer.h.{ind}.mlp.c_fc"
    pref_cproj= f"transformer.h.{ind}.mlp.c_proj"

    sd_origin = {
        "c_fc.weight"  : origin[f"{pref_cfc}.weight"],
        "c_proj.weight": origin[f"{pref_cproj}.weight"]
    }

    sd_VL = {
        f"{cfc}_1"  : VL[f"{pref_cfc}_1"],
        f"{cproj}_1": VL[f"{pref_cproj}_1"],
        f"{cfc}_2"  : VL[f"{pref_cfc}_2"],
        f"{cproj}_2": VL[f"{pref_cproj}_2"]
    }

    return sd_origin, sd_VL

class MLP(nn.Module):
    def __init__(self) :
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class KronyMLP(nn.Module):
    def __init__(self) :
        super().__init__()
        self.c_fc_1    = nn.Parameter(torch.normal(0, 0.02, size=(1536,32)))
        self.c_fc_2    = nn.Parameter(torch.normal(0, 0.02, size=(1, 12)))
        self.gelu    = nn.GELU()
        self.c_proj_1  = nn.Parameter(torch.normal(0, 0.02, size=(32,1536)))
        self.c_proj_2  = nn.Parameter(torch.normal(0, 0.02, size=(12,1)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x @ torch.kron(self.c_fc_1, self.c_fc_2).T
        x = self.gelu(x)
        x = x @ torch.kron(self.c_proj_1, self.c_proj_2).T
        x = self.dropout(x)
        return x


# model init

orig = MLP()
KP = KronyMLP()

# state_dict
orig_sd, VL_sd = get_from_state_dict(0, origin, VL)

# loading state dicts
orig.load_state_dict(orig_sd)
KP.load_state_dict(VL_sd)

# freezing the weights of MLP:
for w in orig.parameters(): w.requires_grad = False


loss = nn.MSELoss()
optimizer = torch.optim.SGD(KP.parameters(), lr=0.01)

for i in range(100):
  x, y = torch.randn(16, n_embd, requires_grad = True, device=device), orig(x)

  f_x = KP(x)
  l = loss(y, f_x)

  print(f"iter {i} loss {l}")

  optimizer.zero_grad()
  l.backward()
  optimizer.step()



"""
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
"""