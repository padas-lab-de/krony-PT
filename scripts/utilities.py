# Plug and Play // Most used scripts.

## Loading a gpt model > torch format



## Loading a kronyPT model. and eval:
import numpy 
from model import 
config_args = dict(
    n_layer=12, 
    n_head=12, 
    n_embd=768,
    vocab_size = 50257,
    block_size = 1024,
    bias = True,
    dim_1 = 3072,
    dim_2 = 384
)

batch_size = 12
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

path = "checkpoints/gpt2-prune-new_init_1_iteration_27900.pt"
sd_krony =  torch.load(path)

for pn,p in list(sd_krony.items()):
	if pn.startswith("module"):
		sd_krony[pn[7:]] = sd_krony.pop(pn)

# the state_dict for Krony is this sd_krony (without the bias)

krony_conf = KronyGPTConfig(**config_args)
krony = KronyGPT(krony_conf)

# Loading the GPTs:

# gpt init
conf = GPTConfig(**config0)
gpt = GPT(conf)
sd1 = gpt.state_dict()
k1  = sd1.keys()



## Loading and testing.



## Eval of a local model.


