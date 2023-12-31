import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from model_origin import *
from transformers import GPT2LMHeadModel

config_args = dict(
n_layer=12, 
n_head=12, 
n_embd=768,
vocab_size = 50257,
block_size = 1024,
bias = True
)

# model init using our configs.
config = GPTConfig(**config_args)
model = GPT(config)
sd = model.state_dict()
sd_keys = sd.keys()
sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

# hf weights
model_hf = GPT2LMHeadModel.from_pretrained("GPT2")
sd_hf = model_hf.state_dict()


# sync-ing the naming ...
sd_keys_hf = sd_hf.keys()
sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
assert len(sd_keys_hf) == len(sd_keys), f"mismatched"
for k in sd_keys_hf:
	if any(k.endswith(w) for w in transposed):
		assert sd_hf[k].shape[::-1] == sd[k].shape
		with torch.no_grad():
			sd[k].copy_(sd_hf[k].t())
	else:
		assert sd_hf[k].shape == sd[k].shape
		with torch.no_grad():
			sd[k].copy_(sd_hf[k])

# data loading / owt
path = 'data/openwebtext/'
train_data = np.memmap(f'{path}train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap(f'{path}val.bin', dtype=np.uint16, mode='r')

# some manual stuff should be done auto, I'm sick of your unprofessionalism Ayoub!
block_size = 1024
batch_size = 12
device = "cuda"
device_type = "cuda"
eval_iters = 200
ctx =  torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

def get_batch(split):
	data = train_data if split == 'train' else val_data
	ix = torch.randint(len(data) - block_size, (batch_size,))
	x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
	y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
	return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

@torch.no_grad()
def estimate_loss():
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

model.to(device)
print("G shit only")
print(estimate_loss())
