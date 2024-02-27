
from transformers import GPT2LMHeadModel, GPT2Config
from model import *
import numpy as np
from model_origin import *

#import lm_eval
import torch

batch_size = 12
block_size = 1024
device = "cuda"
##
path = 'data/openwebtext/'
train_data = np.memmap(f'{path}train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap(f'{path}val.bin', dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

x, y = get_batch("train")

#GPT2 king
gpt2= GPT2LMHeadModel.from_pretrained("gpt2")
sd = gpt2.state_dict()
k = sd.keys()

## GPT torch / normal
config0 = dict(
    n_layer=12, 
    n_head=12, 
    n_embd=768,
    vocab_size = 50257,
    block_size = 1024,
    bias = True,
)

conf = GPTConfig(**config0)

gpt0 = GPT(conf)
gpt0 = gpt0.from_pretrained("gpt2")
sd0  = gpt0.state_dict()

gpt0.to(device)
gpt2.to(device)

r0 = gpt0(x)
r2 = gpt2(x)

transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

for i in range(12):
    print("Batch > \n")
    print(f"{r0[0][i][0,:10]}")
    print(f"{r2[0][i][-1][:10]}")

def hf_gpt_sd(sdd):
    wow = {}
    k1 = [i for i in k if any(i.endswith(hh) for hh in transposed)] 
    k2 = [i for i in k if  not any(i.endswith(hh) for hh in transposed)] 
    for i in k1:
        wow[i] = sdd[i].t()
    for i in k2:
        wow[i] = sdd[i]
    return wow

#woohoo = hf_gpt_sd(sd0)

#model.load_state_dict(woohoo)
#model.save_pretrained('./tt3')

