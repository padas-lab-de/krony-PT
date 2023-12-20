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


device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
# 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True


config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

master_process = True
seed_offset = 0
ddp_world_size = 1

out_dir = 'out'

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

device_type = 'cuda'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# poor man's data loader
data_dir =   'data/shakespeare_char'

train_data = np.memmap(f"{data_dir}/train.bin", dtype=np.uint16, mode='r')
val_data = np.memmap(f'{data_dir}/val.bin', dtype=np.uint16, mode='r')

gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters


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



# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

lr_decay_iters = 1000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

wandb_log = False
warmup_iters = 50
learning_rate = 1e-3

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

print(f"geuss we are here after all")

from model_origin import GPT, GPTConfig
from model import KronyGPT

# GPT conf for both teacher and student net
ckpt_path = 'out-shakespeare-char/ckpt.pt'
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']

model_args = {}

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
# freez the weight of the teacher model:

for n,p in teacher.named_parameters():
    p.requires_grad = False
    


# Loading the student:

ckpt_path_VL = 'out-shakespeare-char/ckpt_VL.pt'
checkpoint_VL = torch.load(ckpt_path_VL, map_location=device)

student = KronyGPT(gptconf)
state_dict_VL = checkpoint_VL['model']
student.load_state_dict(state_dict_VL)

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
teacher.lm_head = teacher.lm_head

###  student:
student_wte = student.transformer.wte
student_wpe = student.transformer.wpe
student_drop = student.transformer.drop
student_h = student.transformer.h
student_ln_f = student.transformer.ln_f
student.lm_head = student.lm_head

## eval function:

eval_iters = 10

@torch.no_grad()
def estimate_loss():
    out = {}
    student.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _ , loss = student(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    student.train()
    return out

# let the game begin:

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

optimizer = student.configure_optimizers(weight_decay , learning_rate, (beta1, beta2), device_type)

for it in range(100):
    lr = get_lr(iter_num)
    batch, target = get_batch(train_data)
    b, t = batch.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    #teacher forward pass
    tok_emb_teacher = teacher_wte(batch)
    pos_emb_teacher = teacher_wpe(pos)
    x_teacher = teacher_drop(tok_emb_teacher + pos_emb_teacher)

    # student forward pass
    tok_emb = student_wte(batch)
    pos_emb = student_wpe(pos)
    x = student_drop(tok_emb + pos_emb)

    l = (x - x_teacher)**2

    for i in range(6):  # for each layer...
        x = student_h[i](x)
        x_teacher = teacher_h[i](x)

        l += (x - x_teacher)**2

    ll  = l.mean()
    print(ll)

    optimizer.zero_grad()
    ll.backward()
    optimizer.step()


'''


# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
#raw_model = model.module if ddp else model # unwrap DDP container if needed
raw_model = model # unwrap DDP container if needed

running_mfu = -1.0

while True:
    ## wtf is this branch about?

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        
        if losses['val'] < 1.45 or always_save_checkpoint:
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
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt_VL_10k.pt'))
     
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        #if ddp: > deleted.
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # new batch
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        #print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

'''