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

if True: # vs code, hiding stuff
    config_args = dict(
        n_layer=12, 
        n_head=12, 
        n_embd=768,
        vocab_size = 50257,
        block_size = 1024,
        bias = True,
    )


    batch_size = 0
    block_size = config_args["block_size"]

    eval_iters = 200 # used in estimate_loss()

    gradient_accumulation_steps = 1
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    ddp = 0
    iter_num = 0  
    cut_the_run = 0

    seed_offset = 0

    device = "cuda"
    device_type = "cuda"
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # this gets updated.
    eval_interval = 0
    learning_rate = 1
    weight_decay = 1
    min_lr =1
    max_iters = 0
    warmup_iters =1
    lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla

    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    decay_lr = True # whether to decay the learning rate

    # leftover from train.py

    backend = 'nccl' # 'nccl', 'gloo', etc.
    out_dir = 'out'

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1


ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

if True:
    #eval_iters = 100 
    path = 'data/openwebtext/'
    train_data = np.memmap(f'{path}train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap(f'{path}val.bin', dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

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

#wandb_log = False
#if wandb_log and master_process:
#    import wandb
#    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
#GPT2 hf chekckpoint:

# Case 1: Normy GPT
print(">>> GPT loading, the teacher")
GPT_state_dict = torch.load("out/GPT2.pt")
conf = GPTConfig(**config_args)
teacher = GPT(conf)
teacher.load_state_dict(GPT_state_dict)
print(f"Loading to GPU")
teacher.to(device)

# Case 2:  Kronecker GPT 
print(">>> KronyGPT loading (the student)")
krony_state_dict = torch.load("checkpoints/prune-small-batch-5-4-12_iteration_8100.pt")
# small cleaning. ddp leftovers.
for pn,p in list(krony_state_dict.items()):
	if pn.startswith("module"):
		krony_state_dict[pn[7:]] = krony_state_dict.pop(pn)

krony_conf = KronyGPTConfig(**config_args)
student = KronyGPT(krony_conf)
student.load_state_dict(krony_state_dict )
student.to(device)

if master_process and False:
    print(f">> Computing the loss over {eval_iters} batches of {batch_size}")
    print(f"Loss for Normyteacher is {estimate_loss(teacher)}")
    print(f">> Computing the loss over {eval_iters} batches of {batch_size}")
    print(f"Loss for student is {estimate_loss(student)}")

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

# Initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = student.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if ddp:
    student = DDP(student, device_ids=[ddp_local_rank])

# wandb related.
eval_only = False # bring this up there
wandb_log = False # .. 

wandb_project  = "hallo"
wandb_run_name = "servus"
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# freezing the weights of the teacher.
with torch.no_grad():
    [p.requires_grad_(False) for _,p in teacher.named_parameters()]

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_student = student.module if ddp else student # unwrap DDP container if needed
running_mfu = -1.0

while iter_num < cut_the_run:
    loss = 0
    lr = get_lr(iter_num) if decay_lr else learning_rate    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#	#if iter_num % eval_interval == 0 and master_process:
    if master_process:
        losses = estimate_loss(student)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr
            })
     
    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            student.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            # Forward, manual, like real men
            b, t = X.size()
            device = X.device
            assert t <= block_size, f"Cannot forward sequence of length {t}, block size is only {block_size}"
            pos = torch.arange(0, t, dtype=torch.long, device=device) 

            # teacher forward pass
            tok_emb = teacher.transformer.wte(X) 
            pos_emb = teacher.transformer.wpe(pos) 
            x_teacher = teacher.transformer.drop(tok_emb + pos_emb)
            
            # student forward pass
            tok_emb = student.module.transformer.wte(X) 
            pos_emb = student.module.transformer.wpe(pos) 
            x = student.module.transformer.drop(tok_emb + pos_emb)

            #print("I am so here", x.shape, x_teacher.shape)

            loss = torch.nn.MSELoss()(x, x_teacher) 
            if master_process:
                print(">>>> iteration ", iter_num)
                print(">>>> first elt", torch.nn.MSELoss()(x, x_teacher))
            #print("I am so here", loss)

            for l in range(len(teacher_h)):
                x           = student.module.transformer.h[l](x)
                x_teacher   = teacher.transformer.h[l](x)
                if master_process:
                    print(">>>> Second elt", torch.nn.MSELoss()(x, x_teacher))
                loss += torch.nn.MSELoss()(x, x_teacher)

            x = student.module.transformer.ln_f(x)
            logits = student.module.lm_head(x)
            entropy_loss  =  F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)/gradient_accumulation_steps
            if master_process:
                print(">>>> Third elt",entropy_loss)
            loss +=  entropy_loss

        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

	#if iter_num > 1 and iter_num % 900 == 0 and master_process:
	#	print(f"Saving the checkpoint at iteration {iter_num}!")
	#	torch.save(student.state_dict(), f"checkpoints/{wandb_run_name}_iteration_{iter_num}.pt")
    iter_num += 1
    local_iter_num += 1
    if iter_num >= max_iters:
        break
if ddp:
	destroy_process_group()

"""
        with ctx:
            device = idx.device
            b, t = idx.size()
            assert t <= block_size, f"Cannot forward sequence of length {t}, block size is only {block_size}"
            pos = torch.arange(0, t, dtype=torch.long, device=device) 

            tok_emb = student.transformer.wte(idx) 
            pos_emb = student.transformer.wpe(pos) 
            x = student.transformer.drop(tok_emb + pos_emb)

            for l in range(len(teacher_h)):
                x = student_h[l](x)
            x = student.transformer.ln_f(x)

            logits = student.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation



while iter_num < cut_the_run:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            student.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            device = idx.device
            b, t = idx.size()
            assert t <= block_size, f"Cannot forward sequence of length {t}, block size is only {block_size}"
            pos = torch.arange(0, t, dtype=torch.long, device=device) 

            tok_emb = student.transformer.wte(idx) 
            pos_emb = student.transformer.wpe(pos) 
            x = student.transformer.drop(tok_emb + pos_emb)

            for l in range(len(teacher_h)):
                x = student_h[l](x)
            x = student.transformer.ln_f(x)

            logits = student.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

        idx, targets = get_batch('train') 
        scaler.scale(loss).backward()
    if master_process == True and iter_num % 20 == 0:
        print()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)

    scaler.step(optimizer)  # step the optimizer and scaler if training in fp16
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    iter_num += 1
    local_iter_num += 1

if ddp:
    destroy_process_group()

"""