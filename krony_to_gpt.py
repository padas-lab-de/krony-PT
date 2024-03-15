import torch
import numpy as np

from model_origin import *
from model import *


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
    
    # gpt2 basic config
    config0 = dict(
        n_layer=12, 
        n_head=12, 
        n_embd=768,
        vocab_size = 50257,
        block_size = 1024,
        bias = True,
    )


check_f = [
            "./OG-checks/4000.pt",
            "./OG-checks/1350.pt",
]

pot = "./check2/768_768_2_iteration_13.pt"

sd_krony =  torch.load(pot)
krony_conf = KronyGPTConfig(**config_args)
krony = KronyGPT(krony_conf)
krony.load_state_dict(sd_krony)    

# gpt init
conf = GPTConfig(**config0)
gpt  = GPT(conf)
sd1  = gpt.state_dict()
k1   = sd1.keys()

# I am loading the old format of kronyPT, namely without the bias. Hence, I have to fill.
sd_k = sd_krony.keys()

l_common = [i for i in k1 if i in sd_k] #common
l        = [i for i in k1 if i not in sd_k]
l_weight = [i for i in l if i.endswith(".weight")]
l_bias   = [i for i in l if not i.endswith(".weight")]

def kron_to_gpt(state_d):
    """
    Converts a KronyPT (GPT with Kroneckers as MLP) to Normal GPT
    """
    wow = {}
    for i in l_common:
        wow[i] = state_d[i]

    # bias:
    for i in l_bias:
        s = i[:-5]+"_bias"
        wow[i] = state_d[s]

    # kroneckers
    for i in l_weight:
        f0 = i[:-7]+"_0"
        f1 = i[:-7]+"_1"
        if "c_fc" in f0:
            m0 = state_d[f0].contiguous()
            m1 = state_d[f1].contiguous()
        else:
            m0 = state_d[f0]
            m1 = state_d[f1]
        s  = torch.kron(m0[0],m1[0])
        for f in range(1, config_args["factors"]):
            s  += torch.kron(m0[f],m1[f])
        wow[i] =  s.t()
    return wow

def hf_gpt_sd(sdd, gpt_keys):
    wow1 = {}
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    #transposed = ['attn.c_attn.weight', 'attn.c_proj.weight' ]
    k1 = [i for i in gpt_keys if any(i.endswith(hh) for hh in transposed)] 
    k2 = [i for i in gpt_keys if  not any(i.endswith(hh) for hh in transposed)] 

    for i in k1:
        wow1[i] = sdd[i].t()
    for i in k2:
        wow1[i] = sdd[i]
    return wow1



from transformers import GPT2LMHeadModel, GPT2Config
model  = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_keys    = model.state_dict().keys()


print("Model conversion")
wow = kron_to_gpt(sd_krony)
w = hf_gpt_sd(wow, gpt2_keys)
model.load_state_dict(w)




#gpt.load_state_dict(wow)

print("Saving, Good luck!")
model.save_pretrained("./hf/13")





"""

x, y = get_batch("train")

print("we going GPU >")
krony.to(device)
gpt.to(device)
model.to(device)
r = krony(x)
r1 = gpt(x)
r2 = model(x)

for i in range(5):
    print("\n>> Batch > \n")
    print(f"{r[0][i][0,:10]}")
    print(f"{r1[0][i][0,:10]}")
    print(f"{r2[0][i][-1][:10]}")

for i in l_weight[:2]:
    f0 = i[:-7]+"_0"
    f1 = i[:-7]+"_1"

    if "c_fc" in f0:
        m0 = sd_krony[f0].contiguous()
        m1 = sd_krony[f1].contiguous()
    else:
        m0 = sd_krony[f0]
        m1 = sd_krony[f1]

    s  = torch.kron(m0[0],m1[0])
    print(sd_krony[f0].shape)
    print(sd_krony[f1].shape)
    print(s.shape)
    print(sd1[i].shape)

for i in l_weight:
    f0 = i[:-7]+"_0"
    f1 = i[:-7]+"_1"
    if "c_fc" in f0:
        m0 = state_d[f0].contiguous()
        m1 = state_d[f1].contiguous()
    else:
        m0 = state_d[f0]
        m1 = state_d[f1]
    s  = torch.kron(m0[0],m1[0])
    for f in range(config_args["factors"]):
        s  += torch.kron(m0[f],m1[f])
    wow[i] =  s

i = "transformer.h.0.mlp.c_proj.weight"
f0 = i[:-7]+"_0"
f1 = i[:-7]+"_1"
m0 = sd_krony[f0].contiguous()
m1 = sd_krony[f1].contiguous()
s  = torch.kron(m0[0],m1[0])
for f in range(config_args["factors"]):
    s  += torch.kron(m0[f],m1[f])


############################################################################################################

"""

"""

# step 2: From  Anrej GPT sd   TO    HF GPT
# load the models to gpu first
#model = None
#print(f"Computing the loss over {eval_iters} batches of 12")
#print(f"Loss for krony with zeros bias >>  {estimate_loss(krony)}")
#print(f"Computing the loss over {eval_iters} batches of 12")
#print(f"Loss for krony with zeros bias >>  {estimate_loss(gpt)}")

############################################################################

> This block was nece. before, when I used to train models with no bias.

tintin = {k: v for k, v in sd_krony.items()}

int_sd = krony.state_dict()
for i in int_sd.keys():
    if i not in sd_krony.keys():
        x = int_sd[i].shape
        tintin[i] = torch.zeros(x)
 = 0
    ...: for i in set(sd.keys()):
    ...:     if i.endswith("_0") or i.endswith("_1"):
    ...:         k = i[:-3]+i[-1]
    ...:         sd[k] = sd.pop(i).unsqueeze(0)
    ...:         #print(i,k)
    ...:     #sd[i[7:]] = sd.pop(i)
"""
