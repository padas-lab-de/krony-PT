from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import torch
from tqdm import tqdm
from model import *
from model_origin import *

import numpy as np  

# init of Krony model:
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

krony_conf = KronyGPTConfig(**config_args)
krony = KronyGPT(krony_conf)
sd_krony = krony.state_dict()
sd_k = sd_krony.keys()

conf = GPTConfig(**config0)
gpt  = GPT(conf)
sd1  = gpt.state_dict()
k1   = sd1.keys()

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
#        s = sd1[i].shape
#        wow[i] = torch.zeros(s)
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



device = "cuda"

#model_id = "openai-community/gpt2"
#model_id = "./models/180000"
#model_id = "./ww3"
#model_id = "distilbert/distilgpt2"


checks = [
            "gold_gold_4_32_iteration_1150.pt",
            "gold_gold_4_32_iteration_1350.pt",
            "gold_gold_4_32_iteration_1450.pt",
            "gold_gold_4_32_iteration_1550.pt",
            "gold_gold_4_32_iteration_1600.pt",
            "gold_gold_4_32_iteration_1650.pt",
            "gold_gold_4_32_iteration_1800.pt",
            "gold_gold_4_32_iteration_1900.pt",
            "gold_gold_4_32_iteration_1950.pt",
            "gold_gold_4_32_iteration_2000.pt",
            "gold_gold_4_32_iteration_2050.pt",
            "gold_gold_4_32_iteration_2100.pt",
            "gold_gold_4_32_iteration_2150.pt",
            "gold_gold_4_32_iteration_2200.pt",
            "gold_gold_4_32_iteration_2350.pt",
            "gold_gold_4_32_iteration_2400.pt",
            "gold_gold_4_32_iteration_2450.pt",
            "gold_gold_4_32_iteration_2500.pt",
            "gold_gold_4_32_iteration_2550.pt",
            "gold_gold_4_32_iteration_2600.pt",
            "gold_gold_4_32_iteration_2650.pt",
            "gold_gold_4_32_iteration_2750.pt",
            "gold_gold_4_32_iteration_2800.pt",
            "gold_gold_4_32_iteration_2850.pt",
            "gold_gold_4_32_iteration_2900.pt",
            "gold_gold_4_32_iteration_2950.pt",
            "gold_gold_4_32_iteration_3000.pt",
            "gold_gold_4_32_iteration_300.pt",
            "gold_gold_4_32_iteration_3050.pt",
            "gold_gold_4_32_iteration_3100.pt",
            "gold_gold_4_32_iteration_3150.pt",
            "gold_gold_4_32_iteration_3200.pt",
            "gold_gold_4_32_iteration_3250.pt",
            "gold_gold_4_32_iteration_3300.pt",
            "gold_gold_4_32_iteration_3350.pt",
            "gold_gold_4_32_iteration_3400.pt",
            "gold_gold_4_32_iteration_3450.pt",
            "gold_gold_4_32_iteration_3500.pt",
            "gold_gold_4_32_iteration_350.pt",
            "gold_gold_4_32_iteration_3550.pt",
            "gold_gold_4_32_iteration_3600.pt",
            "gold_gold_4_32_iteration_3650.pt",
            "gold_gold_4_32_iteration_3700.pt",
            "gold_gold_4_32_iteration_3750.pt",
            "gold_gold_4_32_iteration_3800.pt",
            "gold_gold_4_32_iteration_3850.pt",
            "gold_gold_4_32_iteration_3900.pt",
            "gold_gold_4_32_iteration_3950.pt",
            "gold_gold_4_32_iteration_4000.pt",
            "gold_gold_4_32_iteration_4050.pt",
            "gold_gold_4_32_iteration_4150.pt",
            "gold_gold_4_32_iteration_4200.pt",
            "gold_gold_4_32_iteration_4300.pt",
            "gold_gold_4_32_iteration_4350.pt",
            "gold_gold_4_32_iteration_4400.pt",
            "gold_gold_4_32_iteration_4450.pt",
            "gold_gold_4_32_iteration_4500.pt",
            "gold_gold_4_32_iteration_4550.pt",
            "gold_gold_4_32_iteration_4600.pt",
            "gold_gold_4_32_iteration_4650.pt",
            "gold_gold_4_32_iteration_4700.pt",
            "gold_gold_4_32_iteration_4750.pt",
            "gold_gold_4_32_iteration_4800.pt",
            "gold_gold_4_32_iteration_4850.pt",
            "gold_gold_4_32_iteration_4900.pt",
            "gold_gold_4_32_iteration_4950.pt",
            "gold_gold_4_32_iteration_5100.pt",
            "gold_gold_4_32_iteration_5150.pt",
            "gold_gold_4_32_iteration_5200.pt",
            "gold_gold_4_32_iteration_5250.pt",
            "gold_gold_4_32_iteration_5300.pt",
            "gold_gold_4_32_iteration_5350.pt",
            "gold_gold_4_32_iteration_5450.pt",
            "gold_gold_4_32_iteration_5500.pt",
            "gold_gold_4_32_iteration_550.pt",
            "gold_gold_4_32_iteration_5550.pt",
            "gold_gold_4_32_iteration_5650.pt",
            "gold_gold_4_32_iteration_5950.pt",
            "gold_gold_4_32_iteration_6000.pt",
            "gold_gold_4_32_iteration_850.pt",
            "gold_gold_4_32_iteration_900.pt",
            "gold_gold_4_32_iteration_950.pt" ]

from transformers import GPT2LMHeadModel, GPT2Config
model  = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_keys    = model.state_dict().keys()
model.to(device)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

#krony.to(device)

check_f = [
            "./imp-checks/gold_gold_4_32_iteration_4000.pt",
            "./imp-checks/gold_gold_4_32_iteration_1350.pt",
]

wiki = ['wikitext-103-v1', 'wikitext-2-v1']

potential = []
ppl_s = []

def test_this(dataset, model):
    if dataset == "wiki103":
        test = load_dataset("wikitext", wiki[0], split="test")
    elif dataset == "wiki2":
        test = load_dataset("wikitext", wiki[1], split="test")
    elif dataset == "lambada":
        test = load_dataset("lambada", split="test")
        
    #test = load_dataset("lambada", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = model.config.n_positions
    stride = 1024
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()




results = {}

for path in check_f:
    print(f"\n############################### Model: {path}")
    state_d = torch.load(path, map_location=device)
    krony.load_state_dict(state_d)  
    krony.to(device)


    # converting krony to gpt to hf-gpt 
    wow = kron_to_gpt(state_d)
    w = hf_gpt_sd(wow, gpt2_keys)
    
    model.load_state_dict(w)
    state_d = None

    #ppl_wiki103 = test_this("wiki103", model)
    ppl_wiki103   = test_this("wiki103",   model)
    ppl_wiki2   = test_this("wiki2",   model)
    ppl_lambada = test_this("lambada", model)

    print(ppl_wiki103 ,ppl_wiki2, ppl_lambada)
    results[path[10:]] = [ppl_wiki103, ppl_wiki2, ppl_lambada]
 
for i,j in results.items():
    print(i,j)
