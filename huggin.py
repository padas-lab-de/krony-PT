from transformers import GPT2LMHeadModel
import lm_eval
import torch


model       = GPT2LMHeadModel.from_pretrained("gpt2")
sd_origin   = model.state_dict()
keys_origin = sd_origin.keys()

sd_krony    = torch.load("checkpoints/gpt2-prune-lr-same-all-batch-12.pt")
for pn,p in list(sd_krony.items()):
	if pn.startswith("module"):
		sd_krony[pn[7:]] = sd_krony.pop(pn)
keys_krony  = sd_krony.keys()

print(f"# keys origin: {len(keys_origin)}  Krony:  {len(keys_krony)}")

wow = {}

l =  [i for i in keys_origin if i in keys_krony] # common.

l1       = [i for i in keys_origin if i not in keys_krony ]
l_weight =  [i for i in l1 if i.endswith(".weight")]
l_bias   =  [i for i in l1 if i.endswith(".bias")]

# TODO: check if l_bias + l_weight  = l1

l2 = [i for i in keys_krony  if i not in keys_origin]

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
model_eval = lm_eval.models.huggingface.HFLM(pretrained=model)
result  = lm_eval.evaluator.simple_evaluate(model_eval, tasks=["wikitext"], device=device, batch_size=8)

print(result["results"])

"""
from transformers import GPT2LMHeadModel
import lm_eval
import torch

model       = GPT2LMHeadModel.from_pretrained("gpt2")
lm_eval.tasks.initialize_tasks() 
model_eval = lm_eval.models.huggingface.HFLM(pretrained=model)
result  = lm_eval.evaluator.simple_evaluate(model_eval, tasks=["wikitext"], batch_size=8)
"""