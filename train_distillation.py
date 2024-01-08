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
    lr_decay_iters = 600000 
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


    data_dir =   'data/openwebtext'
    train_data = np.memmap(f"{data_dir}/train.bin", dtype=np.uint16, mode='r')
    val_data = np.memmap(f'{data_dir}/val.bin', dtype=np.uint16, mode='r')
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

    ## eval function:
    eval_iters = 10
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

    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(f"iterations per epoch: {len(train_data) / tokens_per_iter:,}")


    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout)

    model_args['vocab_size'] =  50257
    model_args['bias'] = True

wandb_log = False
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# GPT2 hf chekckpoint:

print(f"Loading GPT2 124M")
conf = GPTConfig(**model_args)
teacher = GPT(conf)
sd0 = torch.load("out/GPT2.pt")
teacher.load_state_dict(sd0)
teacher.to(device)
print(">> Loading DONE, computing loss")
print(f">> Loss is {estimate_loss(teacher)}")

print("Loading KronyGPT with prune init:")
krony_conf = KronyGPTConfig(**model_args)
student = KronyGPT(krony_conf)
sd1 = torch.load("out/GPT2_prune_init.pt")
student.load_state_dict(sd1)
student.to(device)
print(">> Loading DONE, computing loss")
print(f">> Loss: {estimate_loss(student)}")

# Teacher: layering down
teacher_wte = teacher.transformer.wte
teacher_wpe = teacher.transformer.wpe
teacher_drop = teacher.transformer.drop
teacher_h = teacher.transformer.h
teacher_ln_f = teacher.transformer.ln_f
teacher_lm_head = teacher.lm_head

# Student: layering down
student_wte = student.transformer.wte
student_wpe = student.transformer.wpe
student_drop = student.transformer.drop
student_h = student.transformer.h
student_ln_f = student.transformer.ln_f
student_lm_head = student.lm_head

project= "owt"
name= "1 by 1"

#wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
idx, targets = get_batch('train') 
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model =  student # unwrap DDP container if needed
running_mfu = -1.0


iter_num = 0
ddp = 0

# Initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = student.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)


# Freeze lab, here we decide.  Just how you love it.

with torch.no_grad():
    [p.requires_grad_(False) for _,p in teacher.named_parameters()]
    [p.requires_grad_(False) for _,p in student.named_parameters()]

student_keys = list(student.state_dict().keys())

def unfreeze_names(layer: int):
    return [pn for pn in student_keys if any([pn.endswith("0"), pn.endswith("1")]) and f"h.{layer}." in pn]

# Releasing some weights.
s = 8
stops = [4, 8]

for l in range(s,12):
    [p.requires_grad_(True) for pn,p in student.named_parameters() if pn in unfreeze_names(l)]

lrr = 0.01
print(f"mr l mf'ing r is {lrr}")

while 1:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lrr

    """
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        #print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        
        if losses['val'] < 2.8 or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
    """ 

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            student.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            #>> fun happens here:
            device = idx.device
            b, t = idx.size()
            assert t <= block_size, f"Cannot forward sequence of length {t}, block size is only {block_size}"
            pos = torch.arange(0, t, dtype=torch.long, device=device) 
            tok_emb = student.transformer.wte(idx) 
            pos_emb = student.transformer.wpe(pos) 
            x = student.transformer.drop(tok_emb + pos_emb)

            for b in range(len(teacher_h),s):
                x = teacher_h[b](x)
            for l in range(s, len(teacher_h)):
                x = student_h[l](x)
            x = student.transformer.ln_f(x)

            logits = student.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

        idx, targets = get_batch('train') 
        scaler.scale(loss).backward()
        
    if iter_num % 10 == 0:
        print(f"loss at step {iter_num} >> {loss*gradient_accumulation_steps}")

    if iter_num == 50 :
        print(f"unfreezing the other from 4 to {s}")
        for l in range(4,s):
            [p.requires_grad_(True) for pn,p in student.named_parameters() if pn in unfreeze_names(l)]

    if iter_num == 100:
        print(f"unfreezing the other from 0 to 4")
        for l in range(4):
            [p.requires_grad_(True) for pn,p in student.named_parameters() if pn in unfreeze_names(l)]

    if iter_num == 150:
        lrr = 0.001
        print("unfreezing ALL weights")
        [p.requires_grad_(True) for _,p in student.named_parameters()]

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)

    scaler.step(optimizer)  # step the optimizer and scaler if training in fp16
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    iter_num += 1
    local_iter_num += 1
    # termination conditions

    if iter_num > 350:
        break

if ddp:
    destroy_process_group()
