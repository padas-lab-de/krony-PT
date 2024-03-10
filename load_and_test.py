from transformers import GPT2LMHeadModel, GPT2Config
from model import *
import numpy as np
from model_origin import *
import torch

if True:
    config_args = dict(
        n_layer=12, 
        n_head=12, 
        n_embd=768,
        vocab_size = 50257,
        block_size = 1024,
        bias = True,
        dim_1 = 768,
        dim_2 = 768,
        factors = 1 
    )

    batch_size  = 12
    block_size  = config_args["block_size"]
    device      = "cuda"
    device_type = "cuda"
    eval_iters  = 400 

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


sd_krony =  torch.load(f"./imp-checks/gold_gold_4_32_iteration_1350.pt")

"""
k = list(sd_krony.keys())
k1, k2 = k[0], k[-1]
from transformers import GPT2LMHeadModel, GPT2Config
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
sd2 = gpt2.state_dict()
sd_krony[k1] = sd2[k1]
sd_krony[k2] = sd2[k2]
torch.save(sd_krony, "./imp-checks/1350-fresh-emb.pt")
"""

krony_conf = KronyGPTConfig(**config_args)
krony = KronyGPT(krony_conf)
sd   = krony.state_dict()
keys = sd.keys() 

#krony.load_state_dict(sd_krony)
#krony.to(device)
#print("we loss on this b")
#print(estimate_loss(krony))

wow = {}
for i,j in sd_krony.items():
    if i not in keys:
        x = i[:-3]+i[-1]
        j = j.t().unsqueeze(0)
        wow[x] = j


for i in keys:
    if i not in wow.keys():
        wow[i] = sd_krony[i]
        
        
        
        