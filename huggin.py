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
    config0 = dict(
        n_layer=12, 
        n_head=12, 
        n_embd=768,
        vocab_size = 50257,
        block_size = 1024,
        bias = True,
    )

    conf = GPTConfig(**config0)
    gpt0 = GPT(conf)
    gpt0.from_pretrained("gpt2")

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

model  = GPT2LMHeadModel.from_pretrained("./ggg")
sd_origin   = model.state_dict()
keys_origin = sd_origin.keys()

it = 27900
sd_krony =  torch.load(f"checkpoints/gpt2-prune-new_init_1_iteration_{it}.pt")
for pn,p in list(sd_krony.items()):
	if pn.startswith("module"):
		sd_krony[pn[7:]] = sd_krony.pop(pn)
keys_krony  = sd_krony.keys()
krony_conf = KronyGPTConfig(**config_args)
krony = KronyGPT(krony_conf)

#krony_sd = krony.state_dict()
#k_krony = krony_sd.keys() 
#print(f"# keys origin: {len(keys_origin)}  Krony:  {len(keys_krony)}")

wow = {}



l =  [i for i in keys_origin if i in keys_krony] # common.
l1       = [i for i in keys_origin if i not in keys_krony ]
l_weight =  [i for i in l1 if i.endswith(".weight")]
l_bias   =  [i for i in l1 if i.endswith(".bias")]

# TODO: check if l_bias + l_weight  = l1, DONE, passed!

l2 = [i for i in keys_krony  if i not in keys_origin]

"""
# common keys
for i in l:
    if sd_krony[i].shape == sd_origin[i].shape:
        wow[i] = sd_krony[i]
    else:
        wow[i] = sd_krony[i].transpose(1,0)

# kroneckers:  @weight
for i in l_weight:
    pref = i[:-7]
    f0 = i[:-7]+"_0_0"
    f1 = i[:-7]+"_0_1"
    m0 = sd_krony[f0]
    m1 = sd_krony[f1]
    wow[i] = torch.kron(m0,m1).transpose(1,0)

# bias
so_far = wow.keys()
bias = [i for i in keys_origin if i not in so_far]

for b in bias:
    s = sd_origin[b].shape
    wow[b] = torch.zeros(s)

model.load_state_dict(wow)

device = "cuda:0"
model.to(device)
lm_eval.tasks.initialize_tasks() 

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model_eval = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer = tokenizer)
result  = lm_eval.evaluator.simple_evaluate(model_eval, tasks=["wikitext"], device=device, batch_size=8)
print(result["results"])

# TODO
# > 
# > Train with bias included.
# > Train again on 2 factors with / small models.
# > Try reshaping ? what did you mean here?



from transformers import GPT2LMHeadModel
import lm_eval
import torch

model       = GPT2LMHeadModel.from_pretrained("gpt2")
lm_eval.tasks.initialize_tasks() 
model_eval = lm_eval.models.huggingface.HFLM(pretrained=model)
result  = lm_eval.evaluator.simple_evaluate(model_eval, tasks=["wikitext"], batch_size=8)
"""
