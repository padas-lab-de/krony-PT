"""
Distillation, layer wise decompostion

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext


import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import torch.nn.functional as F

from model_origin import GPT, GPTConfig
from model import KronyGPT
from config.train_shakespeare_char import *


import wandb

device = 'cuda'
device_type = 'cuda'
dtype = 'bfloat16'   # what is the diff between bfloat16 and float16?
compile = True

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]

exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

master_process = True
seed_offset = 0
ddp_world_size = 1


torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# poor man's data loader
data_dir =   'data/shakespeare_char'

train_data = np.memmap(f"{data_dir}/train.bin", dtype=np.uint16, mode='r')
val_data = np.memmap(f'{data_dir}/val.bin', dtype=np.uint16, mode='r')

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size

print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(f"iterations per epoch: {len(train_data) / tokens_per_iter:,}")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


eval_iters = 50

## eval function:
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _ , loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# when you make it work, then fucking log this shit here log this shit here
#if wandb_log and master_process:
#    import wandb
#    wandb.init(project=wandb_project, name=wandb_run_name, config=config)



# GPT conf for both teacher and student net
ckpt_path = 'out-shakespeare-char/ckpt.pt'
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']


model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, vocab_size=None, dropout=dropout)

for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

block_size = model_args["block_size"]
gptconf = GPTConfig(**model_args)

# Loading the teacher:]

teacher = GPT(gptconf)
state_dict = checkpoint['model']

unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

teacher.load_state_dict(state_dict)



# Loading the student:

ckpt_path_VL = 'out-shakespeare-char/ckpt_VL.pt'
checkpoint_VL = torch.load(ckpt_path_VL, map_location=device)

student = KronyGPT(gptconf)
state_dict_VL = checkpoint_VL['model']
student.load_state_dict(state_dict_VL)

# freeze the weight of the student/teacher.
# student weights will updated gradually.

for n,p in teacher.named_parameters():
    p.requires_grad = False
    
for n,p in student.named_parameters():
    p.requires_grad = False

print(f"loading the homies to(device) ")
student.to(device)
teacher.to(device)

### I need to layer down all layers so I have more control when iterating:
###  teacher:
teacher_wte = teacher.transformer.wte
teacher_wpe = teacher.transformer.wpe
teacher_drop = teacher.transformer.drop
teacher_h = teacher.transformer.h
teacher_ln_f = teacher.transformer.ln_f
teacher_lm_head = teacher.lm_head

###  student:
student_wte = student.transformer.wte
student_wpe = student.transformer.wpe
student_drop = student.transformer.drop
student_h = student.transformer.h
student_ln_f = student.transformer.ln_f
student_lm_head = student.lm_head

# let the game begin:

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

print(f"wandb magic gon happen")
# wandb.init(project=wandb_project, name=wandb_run_name, config=config)



def freeze_layer(layer_num: int, freeze: bool):
    for n,p in student_h[layer_num].named_parameters(): 
        if n.endswith("_1") or n.endswith("_2"):
            p.requires_grad = freeze 

for i in range(6):
    freeze_layer(i, True)

optimizer = student.configure_optimizers(weight_decay , learning_rate, (beta1, beta2), device_type)

eval_interval  = 100

for iter_num in range(2000):
    lr = get_lr(iter_num)
    #lr = 0.001
    batch, target = get_batch(train_data)
    b, t = batch.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    #teacher and student, tokens encoding.
    tok_emb = student_wte(batch)
    pos_emb = student_wpe(pos)
    x = student_drop(tok_emb + pos_emb)


    #l = 0 
    distill_layer = iter_num // 1000
    

    l = 0 
    for i in range(distill_layer + 1):
        x = student_h[i](x)
        x_teacher = teacher_h[i](x)
        l += F.mse_loss(x, x_teacher)
        
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

    if iter_num % eval_interval == 0 :
        losses = estimate_loss(student)
        print(f">>>> step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f">> Distillation loss: {l}, learning rate {lr}")

        #wandb.log({
        #    "train/loss": losses['train'],
        #    "val/loss": losses['val'],
        #    "distill loss": l,
        #})


'''
    if iter_num % 1000 == 0:
        cfc = f""
        cproj = f"" 

        if iter_num == 0:
            # unfreeze layer 0
            pass
        else:
            # freeze past layer
            pass
            # unfreeze curret layer

'''
names = list(checkpoint_VL["model"])
n1, n2 = names[0], names[1]

sd_student = student.state_dict()
x1, x2 = sd_student[n1], sd_student[n2]

y1, y2 = state_dict_VL[n1], state_dict_VL[n2]

orig = state_dict[f"{n1[:-2]}.weight"]

### step 1: inits:

### distillation 1 by 1

### model post-training (fine-tuning)               

"""


model-1:

for i in range(6):
    model 2 re-eval:
"""