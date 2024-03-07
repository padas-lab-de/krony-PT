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
        dim_2 = 768
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

    config0 = dict(
        n_layer=12, 
        n_head=12, 
        n_embd=768,
        vocab_size = 50257,
        block_size = 1024,
        bias = True,
    )


check_f = [
            "./imp-checks/gold_gold_4_32_iteration_4000.pt",
            "./imp-checks/gold_gold_4_32_iteration_1350.pt",
]

sd_krony =  torch.load(check_f[0])

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
        f0 = i[:-7]+"_0_0"
        f1 = i[:-7]+"_0_1"
        m0 = state_d[f0]
        m1 = state_d[f1]
        wow[i] = torch.kron(m0,m1)
    return wow


def hf_gpt_sd(sdd, gpt_keys):
    wow1 = {}
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
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

wow = kron_to_gpt(sd_krony)
w = hf_gpt_sd(wow, gpt2_keys)

gpt.load_state_dict(wow)
model.load_state_dict(w)
model.save_pretrained('./models/4000')

print("done - Good luck!")

"""
x, y = get_batch("train")

#model.to(device)
krony.to(device)
gpt.to(device)

#r2 = model(x)
r = krony(x)
r1 = gpt(x)

for i in range(5):
    print("\n>> Batch > \n")
    print(f"{r[0][i][0,:10]}")
    print(f"{r1[0][i][0,:10]}")
#    print(f"{r2[0][i][-1][:10]}")

# step 2: From  Anrej GPT sd   TO    HF GPT
# load the models to gpu first

krony.to(device)
print(f"Computing the loss over {eval_iters} batches of 12")
print(f"Loss for krony with zeros bias >>  {estimate_loss(krony)}")

print(f"Computing the loss over {eval_iters} batches of 12")
print(f"Loss for krony with zeros bias >>  {estimate_loss(gpt)}")


############################################################################

> This block was nece. before, when I used to train models with no bias.

tintin = {k: v for k, v in sd_krony.items()}

int_sd = krony.state_dict()
for i in int_sd.keys():
    if i not in sd_krony.keys():
        x = int_sd[i].shape
        tintin[i] = torch.zeros(x)
 
"""