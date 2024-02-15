"""
Evaluation / Small lab to init fast. 
* I want to load
* And generate
* And even, eval here.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import tiktoken

from model_origin import *
from model import *


from eval_wrapper import *

if True:
    # put some vars here.
    config_args = dict(
        n_layer=12, 
        n_head=12, 
        n_embd=768,
        vocab_size = 50257,
        block_size = 1024,
        bias = True,
    )

    block_size = config_args["block_size"]
    device = "cuda"
    device_type = "cuda"
    eval_iters = 30
    batch_size = 32

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



print("Loading the model, hun!")
conf = GPTConfig(**config_args)
model = GPT(conf)
checkpoint = torch.load("out/GPT2.pt")
model.load_state_dict(checkpoint)
model.to(device)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# the mf'ing model
gen = CustomGeneration(model, encode, config_args)



#from lm_eval.api.model import LM
import lm_eval
from lm_eval.api.instance import Instance


gen_until = False
if gen_until:
    input_strings = [
        "Explain the theory of relativity.",
        "Describe the life cycle of a butterfly.",
        "Outline the process of photosynthesis.",
        "Discuss the history of the internet.",
        "Elaborate on the advancements in AI technology."
    ]

    generation_params = {
        "max_new_tokens": 100,
        "temperature": 0.8,
        "top_k": 200,
        "until": ["\n\n", "."],
        "max_gen_toks": 128
    }

    until = [
        Instance(
            request_type="generate_until",
            doc={"question": input_strings[i]},
            arguments=(input_strings[i], generation_params),
            idx=i
        ) for i in range(5)
    ]

    # generate_until
    with torch.no_grad():
        with ctx:
            o = gen.generate_until(until)
            for i in o:
                print(decode(i[0].tolist()))
                print('\n|---------------|\n')


loglike = True
if loglike:
    log_ins = [
        ("The capital of France is", "Paris"),
        ("Einstein is known for his theory of", "Relativity"),
        ("The largest planet in the Solar System is", "Jupiter"),
        ("The novel '1984' was written by", "George Orwell"),
        ("The process of converting light energy into chemical energy in plants is called", "Photosynthesis")
    ]

    # Creating 5 loglikelihood instances with the specified input and target strings
    loglikelihood_instances = [
        Instance(
            request_type="loglikelihood",
            doc={},  # Empty doc as no additional context is needed
            arguments = log_ins[i],
            idx=i + 10  # Indexes starting from 10 to avoid overlap with previous instances
        ) for i in range(5)
    ]

roll = False 

if roll:
    roll_ins = [
        "The theory of evolution was proposed by Charles Darwin.",
        "Quantum mechanics is a fundamental theory in physics.",
        "Shakespeare wrote many famous plays including Hamlet and Macbeth.",
        "The Great Wall of China is one of the seven wonders of the world.",
        "Artificial Intelligence has significant impacts on various industries."
    ]

    # Creating 5 loglikelihood_rolling instances with the specified input strings
    loglikelihood_rolling_instances = [
        Instance(
            request_type="loglikelihood_rolling",
            doc={},  # Empty doc as no additional context is needed
            arguments=(roll_ins[i],),
            idx=i + 15  # Indexes starting from 15 to avoid overlap with previous instances
        ) for i in range(5)
    ]



from lm_eval import tasks, evaluator



for (s0,s1) in log_ins:
    s1 = s0 + " " + s1
    s0, s1 = torch.tensor([encode(s0)]).to(device), torch.tensor([encode(s1)]).to(device)
    r = s1.shape[1] - s0.shape[1]
    x0, x1 = s1[:,:-1], s1[:,1:]
    lx, l = model(x0, x1)
    t1, t2= lx[:,-r:,:], x1[:,-r:]
    
    lxx = torch.gather(lx, 2, x1.unsqueeze(-1)).squeeze(-1)

    #print("Input only", len(s0)) 
    #print("Input + Output: ", len(s1)) 
    #print("New output: ", len(s1)-len(s0)) 
    
    


"""
Sum up:
* x, y =  get_batch("train") 
    * have the same shapa batch_size X block_size
    * x[:,1:] == y[:,:-1] True. (every y = x + 1 token at the end.)

* How is the loss computed?
    * logits, _ = model(x,y) 
    * logits has size [32, 1024, 50k] ==> for each token, we have 


start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# this should be handled internally with Instances.

cfn = {"max_new_tokens" : 100, "temperature" : 0.8, "top_k" : 200}
x,y = get_batch("train")
logits, loss = model(x,y)
print(enc.eot_token)

xx = logits.view(-1, logits.size(-1))  # [32*1024  x 50k]
yy = y.view(-1)                        # [32*1024]

print(f"the shapes {xx.shape}, {yy.shape}")

print("The big fucking loss >>\n\n")
print(F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1))

n  = 30
x1 = xx[:n,:]
x2 = F.log_softmax(x1, dim = 1)
y1 = yy[:n]

print("the Mini loss is", F.cross_entropy(x1, y1, ignore_index=-1))
print(x1, y1)

s = 0
with torch.no_grad():
    for i in range(len(y1)):
        print(x2[i,y1[i]])
        s+= x2[i,y1[i]].cpu()
        #s+= - np.log(x1[i,y1[i]].cpu())
    
print(s, s/n)

A small note on get_batch():

* x,y are both batch_size x block_size 
* y is offset by one token. x = [x1, x2, x3,.., xN] and y  = [x2, x3,.., xN, x{N+1}]
* So, what does, model(X,Y) do?
* 
"""
