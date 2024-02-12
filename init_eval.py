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

# TODO: this has to move to a proper file. 
# Remove all the occurencess of hun from this file. bitte!

num_samples = 5
max_new_tokens = 100
temperature = 0.8
top_k = 200

# change this hun
start = "The founder of SpaceX and Tesla is " 
print(f"Now we generate {num_samples} samples of the following prompt: {start}")

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

gen = CustomGeneration(model, encode, config_args)

start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])


# this should be handled internally with Instances.

cfn = {"max_new_tokens" : 100, "temperature" : 0.8, "top_k" : 200}
likelihood_reqs = [
   ("Who is Elon Musk", cfn),
   ("Who owns Google? Good question, the answer is", cfn)
] 

# run generation
if False: # this is self.generate_until 
	with torch.no_grad():
		with ctx:
			for k in range(1):
				y = gen.generate_until(likelihood_reqs)
				for i in y:
					print(decode(i[0].tolist()))
			print('|---------------|')


# now we test loglikelihood
with torch.no_grad():
	with ctx:
		for k in range(0):
			y = gen.generate_until(likelihood_reqs)
			for i in y:
				print(decode(i[0].tolist()))
		print('|---------------|')


x,y = get_batch("train")
logits, loss = model(x,y)

print(x,y)
print(enc.eot_token)

# TODO
# install lm-eval
# play with it. smth that only requires generate_until
#

"""
print(logits.shape)
print(logits.view(-1, logits.size(-1)).shape)
print(y.view(-1).shape)
xx = logits.view(-1, logits.size(-1))
yy = y.view(-1)

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
