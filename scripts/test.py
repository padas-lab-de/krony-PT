"""
This is a lab, literally. I test and verify correctness of code here.

1. I'm checking if the Optimizer is working as I want. Seems ok.
2. Data stuff.

"""


from regex import P
import torch
from model import KronyGPTConfig, KronyGPT


if True:  # stuff I don't really care for the moment: 
    gradient_accumulation_steps = 5 * 4 # used to simulate larger batch sizes
    batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size = 1024
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    device_type = 'cuda' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

config_args = dict(n_layer=12, 
                   n_head=12, n_embd=768, vocab_size =  50257, 
                   block_size = 1024, bias = True)

# stuff related to the optimized.
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
device="cuda"

print("Loading ckpt")
sd = torch.load("out/GPT2_prune_init.pt", map_location=device)

config = KronyGPTConfig(**config_args)
model  = KronyGPT(config)
model.load_state_dict(sd)
model.to(device)

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

sdd = model.state_dict()
nms = list(sdd.keys())



## now we investigate the data:
import numpy as np
import os 

dataset = 'openwebtext'

data_dir = os.path.join('data', dataset)

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), 
                       dtype=np.uint16, mode='r')

val_data = np.memmap(os.path.join(data_dir, 'val.bin'), 
                     dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


X,Y = get_batch("train")

max_iters = 5000

if 1:
    print(">>>> Some data stats")
    tpi = gradient_accumulation_steps * 4 * batch_size * block_size
    print(f"tokens per iteration will be: {tpi:,}")
    print(f"Train data size {len(train_data):_}")
    print(f"To see all data we need {len(train_data)/tpi:.1f} iterations")
    print(f"With {max_iters} iterations, we are going to see {max_iters*tpi/len(train_data):.3f} %")


