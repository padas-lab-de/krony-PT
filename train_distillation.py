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
from model import KronyGPT, KronyGPTConfig

import wandb


if True:
    out_dir = 'out'
    eval_interval = 2000
    log_interval = 1
    eval_iters = 200
    eval_only = False # if True, script exits right after the first eval
    always_save_checkpoint = True # if True, always save a checkpoint after each eval
    init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
    # wandb logging
    wandb_log = False # disabled by default
    wandb_project = 'owt'
    wandb_run_name = 'gpt2' # 'run' + str(time.time())
    # data
    dataset = 'openwebtext'
    gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
    batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size = 1024
    # model
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    learning_rate = 6e-4 # max learning rate
    max_iters = 600000 # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 2000 # how many steps to warm up for
    lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
    min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True # use PyTorch 2.0 to compile the model to be faster

    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging

    master_process = True
    seed_offset = 0
    ddp_world_size = 1

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    device_type = "cuda"
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)


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

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

model_args['vocab_size'] =  50304
model_args['bias'] = True

### Case 1: Normy GPT
conf = GPTConfig(**model_args)
normy_gpt = GPT(conf)

### Loading The Checkpoints
checkpoint = torch.load("out/GPT2.pt")
normy_gpt.load_state_dict(checkpoint)


## Normy Loss
normy_gpt.to(device)
print(f"Loss for NormyGPT is {estimate_loss(normy_gpt)}")

# Case 2:  Kronecker & VL init.
print("KronyGPT with VL init:")
sd = torch.load("out/GPT2_VL11.pt")
print(">> Loading DONE")

# this would would be the same for all.
krony_conf = KronyGPTConfig(**model_args)
krony_gpt = KronyGPT(krony_conf)
krony_gpt.load_state_dict(sd)
krony_gpt.to(device)


# GPT conf for both teacher and student net
checkpoint = torch.load(' ', map_location=device)
checkpoint_model_args = checkpoint['model_args']


model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, vocab_size=None, dropout=dropout)

for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

# Teacher: Loading
block_size = model_args["block_size"]
gptconf = GPTConfig(**model_args)
teacher = GPT(gptconf)
state_dict = checkpoint['model']

teacher.load_state_dict(state_dict)

# Teacher: layering down

teacher_wte = teacher.transformer.wte
teacher_wpe = teacher.transformer.wpe
teacher_drop = teacher.transformer.drop
teacher_h = teacher.transformer.h
teacher_ln_f = teacher.transformer.ln_f
teacher_lm_head = teacher.lm_head

# Teacher: Freezing the weights

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

###  teacher:

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