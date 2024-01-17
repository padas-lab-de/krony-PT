"""
1. Load GPT2 124M and a re-trianed KroneckerGPT 95M with less than 1% data.
2. Test it on different benchmarks. Using: https://github.com/EleutherAI/lm-evaluation-harnessA
"""
  
import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F

from model_origin import *
from model import *


# put some vars here.
config_args = dict(
	n_layer=12, 
	n_head=12, 
	n_embd=768,
	vocab_size = 50257,
	block_size = 1024,
	bias = True,
)

batch_size = 12
block_size = config_args["block_size"]
device = "cuda"
device_type = "cuda"

eval_iters = 200 # used in estimate_loss()


# data loader.
path = 'data/openwebtext/'
train_data = np.memmap(f'{path}train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap(f'{path}val.bin', dtype=np.uint16, mode='r')
def get_batch(split):
	data = train_data if split == 'train' else val_data
	ix = torch.randint(len(data) - block_size, (batch_size,))
	x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
	y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
	return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

ctx =  torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
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
print("GPT loading")
GPT_state_dict = torch.load("out/GPT2.pt")
conf = GPTConfig(**config_args)
GPT = GPT(conf)
GPT.load_state_dict(GPT_state_dict )
print(f"Loading to GPU")
GPT.to(device)
print(f"Computing the loss over {eval_iters} batches of 12")
print(f"Loss for NormyGPT is {estimate_loss(GPT)}")

# Case 2:  Kronecker GPT 
print("KronyGPT 1st Loading")
krony_state_dict = torch.load("checkpoints/gpt2-prune-lr-same-all-batch-12.pt")

# small cleaning. ddp leftovers.
for pn,p in list(krony_state_dict.items()):
	if pn.startswith("module"):
		krony_state_dict[pn[7:]] = krony_state_dict.pop(pn)

krony_conf = KronyGPTConfig(**config_args)
KronyGPT = KronyGPT(krony_conf)
KronyGPT.load_state_dict(krony_state_dict )

print("Loading to GPU")
KronyGPT.to(device)
print(f"Computing the loss over {eval_iters} batches of 12")
print(f"Loss for KronyGPT with VL init is {estimate_loss(KronyGPT)}")



# Case 2:  Kronecker GPT 
print("KronyGPT 2nd Loading")
krony_state_dict_2 = torch.load("checkpoints/gpt2-VL-lr-kdjfkdj-same-all-batch-1-warmup-1k.pt")

# small cleaning. ddp leftovers.
for pn,p in list(krony_state_dict.items()):
	if pn.startswith("module"):
		krony_state_dict_2[pn[7:]] = krony_state_dict_2.pop(pn)

KronyGPT.load_state_dict(krony_state_dict )

#print("Loading to GPU")
#KronyGPT.to(device)
print(f"Computing the loss over {eval_iters} batches of 12")
print(f"Loss for KronyGPT2 with VL init is {estimate_loss(KronyGPT)}")

