"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

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
#from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import KronyGPTConfig, KronyGPT
import matplotlib.pyplot as plt

if True:
    cut_the_run = 0
    init_name = "hallo"

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
    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True # use PyTorch 2.0 to compile the model to be faster

    # this gets updated.
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
    dim1 = 17
    dim2 = 177

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

torch.manual_seed(177 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

device_type = 'cuda' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
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

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init 
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, dim_1=dim1, dim_2=dim2
                  ) # start with model_args from command line

## params loading scratch / resume of gpt2
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = KronyGPTConfig(**model_args)
    model = KronyGPT(gptconf)
else:
    print(f"Resuming training from {init_name}")
    #ckpt_path = os.path.join(out_dir, init_name)
    checkpoint = torch.load(init_name, map_location=device)

    for pn,p in list(checkpoint.items()):
        if pn.startswith("module"):
            checkpoint[pn[7:]] = checkpoint.pop(pn)

    # the checkpoints I'm saving are only weights. I might change this behavior later.
    #checkpoint_model_args = checkpoint['model_args']
    #for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    #    model_args[k] = checkpoint_model_args[k]

    # create the model
    #model_args["dim1"] = dim1
    #model_args["dim2"] = dim2
    model_args['vocab_size'] = 50257
    model_args['bias'] = True

    gptconf = KronyGPTConfig(**model_args)
        
    model = KronyGPT(gptconf)
    model.load_state_dict(checkpoint)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# I change the optimized as follows: 
# it was usually the learning_rate, now it's min_lr 
# why: I keep the pre-trained weights lr as the min, and the new ones, 
# I changed the learning rate manually. 
# optmizer.group_params['lr'] = get_lr(it)

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

checkpoint = None
compile = False 

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
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

# adamw optimizer
if master_process:
    print("\n >>>> Some data stats >>>> \n")
    print(f"ddp_world_size {ddp_world_size}")
    print(f"gradient blabla {gradient_accumulation_steps}")
    tpi = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tpi:,}")
    print(f"Train data size {len(train_data):_}")
    print(f"To see all data we need {len(train_data)/tpi:_} iterations")
    print(f"In {cut_the_run} iters. we are going to see {cut_the_run*tpi/len(train_data):.3f} % of the data")
    print("\n >>>> Some data stats >>>> \n")

    print(">>>>> Training is starting now, here is some stats:")
    print("batch size",    batch_size) 
    print("weight_decay",  weight_decay)  
    print("learning_rate", learning_rate) 
    print("weight_decay",  weight_decay)  
    print("min_lr",        min_lr)        
    print("max_iters",     max_iters)     
    print("warmup_iters",  warmup_iters)  
    print("lr_decay_iters",lr_decay_iters)

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

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

while iter_num < cut_the_run:
# determine and set the learning rate for this iteration
	lr = get_lr(iter_num) if decay_lr else learning_rate
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	if iter_num % eval_interval == 0 and master_process:
		losses = estimate_loss()
		print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
		if wandb_log:
			wandb.log({
				"iter": iter_num,
				"train/loss": losses['train'],
				"val/loss": losses['val'],
				"lr": lr
				#"mfu": running_mfu*100, # convert to percentage
			})

	if iter_num == 0 and eval_only:
		break

# forward backward update, with optional gradient accumulation to simulate larger batch size
# and using the GradScaler if data type is float16
# apparently, we always use a gradient accumulation

	for micro_step in range(gradient_accumulation_steps):
		if ddp:
			model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
		with ctx:
			logits, loss = model(X, Y)
			loss = loss / gradient_accumulation_steps 
			# scale the loss to account for gradient accumulation
		
		# immediately async prefetch next batch while model is 
		# doing the forward pass on the GPU >> investigate this in detail, how does it happen.
		# new batch
		X, Y = get_batch('train')
		# backward pass, with gradient scaling if training in fp16
		scaler.scale(loss).backward()

# clip the gradient
	if grad_clip != 0.0:
		scaler.unscale_(optimizer)
		torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

	scaler.step(optimizer)  # step the optimizer and scaler if training in fp16
	scaler.update()
	optimizer.zero_grad(set_to_none=True)


	if iter_num > 1 and iter_num % 900 == 0 and master_process:
		print(f"Saving the checkpoint at iteration {iter_num}!")
		torch.save(model.state_dict(), f"checkpoints/{wandb_run_name}_iteration_{iter_num}.pt")

	iter_num += 1
	local_iter_num += 1

	if iter_num >= max_iters:
		break

if master_process:
    print("\n >>>> Some data stats >>>> \n")
    print(f"ddp_world_size {ddp_world_size}")
    print(f"gradient blabla {gradient_accumulation_steps}")
    tpi = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tpi:,}")
    print(f"Train data size {len(train_data):_}")
    print(f"To see all data we need {len(train_data)/tpi:_} iterations")
    print(f"In {cut_the_run} iters. we are going to see {cut_the_run*tpi/len(train_data):.3f} % of the data")
    print("\n >>>> Some data stats >>>> \n")

    print(">>>>> Training is starting now, here is some stats:")
    print("batch size",    batch_size) 
    print("weight_decay",  weight_decay)  
    print("learning_rate", learning_rate) 
    print("weight_decay",  weight_decay)  
    print("min_lr",        min_lr)        
    print("max_iters",     max_iters)     
    print("warmup_iters",  warmup_iters)  
  
if ddp:
	destroy_process_group()


