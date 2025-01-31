
import torch
import numpy as np

from model import *
import sys

# how to use:
# python owt_eval.py ./path_to_pt_file.pt dim1 dim2 factors

path_checkpoint = sys.argv[1] 
print("loading the checkpoint >> ", path_checkpoint)
sd_krony =  torch.load(path_checkpoint)


# infering the dims from the shape of the c_proj of the first layer.
c_proj = sd_krony["transformer.h.0.mlp.c_proj_0"].shape
dim1    = c_proj[2] 
dim2    = c_proj[1] 
factors = c_proj[0] 

config_args = dict(
    n_layer=12, 
    n_head=12, 
    n_embd=768,
    vocab_size = 50257,
    block_size = 1024,
    bias = True,
    dim_1 = dim1,
    dim_2 = dim2, 
    factors = factors
)

batch_size = 32
block_size = config_args["block_size"]
device = "cuda"
device_type = "cuda"
eval_iters = 200 

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

krony_conf = KronyGPTConfig(**config_args)
krony = KronyGPT(krony_conf)
krony.load_state_dict(sd_krony)    
krony.to(device)

print(estimate_loss(krony))

