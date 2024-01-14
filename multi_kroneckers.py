
import torch
from model import  KronyGPT, KronyGPTConfig


config_args = dict(n_layer=12, n_head=12, n_embd=768, vocab_size =  50257, block_size = 1024, bias = True, factors=2)


config = KronyGPTConfig(**config_args)
model  = KronyGPT(config)

sd = model.state_dict()
nms = list(sd.keys())

h0 = nms[2:18]


[print(pn, p.shape) for pn, p in sd.items() if pn in h0]