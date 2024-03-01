from transformers import GPT2LMHeadModel, GPT2Config
from model import *
import numpy as np
from model_origin import *

import lm_eval
import torch

#config = GPT2Config.from_pretrained("gpt2")
#config.attn_pdrop = 0.0
#config.embd_pdrop = 0.0
#config.resid_pdrop = 0.0
#model1 = GPT2LMHeadModel(config)

if True:
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


sd_krony =  torch.load(f"checkpoints/gpt2-prune-new_init_1_iteration_27900.pt")
for pn,p in list(sd_krony.items()):
	if pn.startswith("module"):
		sd_krony[pn[7:]] = sd_krony.pop(pn)

k_krony = sd_krony.keys()
krony_conf = KronyGPTConfig(**config_args)
krony = KronyGPT(krony_conf)

# Loading the GPTs:
if True:
    config0 = dict(
        n_layer=12, 
        n_head=12, 
        n_embd=768,
        vocab_size = 50257,
        block_size = 1024,
        bias = True,
    )

    conf = GPTConfig(**config0)
    gpt1 = GPT(conf)
    sd1 = torch.load("out/GPT2.pt")
    gpt1.load_state_dict(sd1)


sd1 = gpt1.state_dict()
k1 = sd1.keys() 

# HF model:
model  = GPT2LMHeadModel.from_pretrained("gpt2")
sd     = model.state_dict()
k      = sd.keys()

# from normal GPT to HF:
assert set(k1) == set(k), "Andrej implementation doesn't matche HF-GPT2"
miss = [i for i in k if not torch.equal(sd[i], sd1[i])]

transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

match_test = []
for i in k:
    if any(i.endswith(gg) for gg in transposed):
        x = torch.equal(sd[i].t(), sd1[i])
    else:
        x = torch.equal(sd[i], sd1[i])
    match_test.append(x)
    

# step 1: From Krony GPT sd    TO    Anrej GPT sd 
sd_krony = krony.state_dict()
sd_k = sd_krony.keys()

l_common = [i for i in k1 if i in sd_k] #common

l  = [i for i in k1 if i not in sd_k]
l_weight = [i for i in l if i.endswith(".weight")]
l_bias   = [i for i in l if not i.endswith(".weight")]

def kron_to_gpt(state_d):
    wow = {}
    for i in l_common:
        wow[i] = state_d[i]

    # bias:
    for i in l_bias:
        s = sd1[i].shape
        wow[i] = torch.zeros(s)
    # kroneckers
    for i in l_weight:
        f0 = i[:-7]+"_0_0"
        f1 = i[:-7]+"_0_1"
        m0 = state_d[f0]
        m1 = state_d[f1]
        wow[i] = torch.kron(m0,m1)
    return wow

wow = kron_to_gpt(sd_krony)

# step 2: From  Anrej GPT sd   TO    HF GPT

def hf_gpt_sd(sdd):
    wow1 = {}

    k1 = [i for i in k if any(i.endswith(hh) for hh in transposed)] 
    k2 = [i for i in k if  not any(i.endswith(hh) for hh in transposed)] 

    for i in k1:
        wow1[i] = sdd[i].t()
    for i in k2:
        wow1[i] = sdd[i]

    return wow1
    
sd_f = hf_gpt_sd(sd1)    




model.load_state_dict(sd_f)
model.save_pretrained('./tt1')



"""

model.save_pretrained('./my_model_directory')

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import lm_eval
import torch

device = "cuda:0"

tokenizer1  = GPT2Tokenizer.from_pretrained("gpt2")
model       = GPT2LMHeadModel.from_pretrained("./tt1")

model.to(device)

lm_eval.tasks.initialize_tasks() 

model_eval = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer = tokenizer1)
result  = lm_eval.evaluator.simple_evaluate(model_eval, tasks=["wikitext"], batch_size=8, device = device)

print(result["results"])
"""
