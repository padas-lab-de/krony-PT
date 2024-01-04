'''
 This scripts loads the GPT2 HuggingFace checkpoints.
 if you want replicate, create an out directory first -- where the checkpoint will be stored.
'''


import torch
from transformers import GPT2LMHeadModel
from model_origin import GPTConfig, GPT


model_type = 'gpt2'
config_args = dict(n_layer=12, n_head=12, n_embd=768, vocab_size =  50257, block_size = 1024, bias = True)


config = GPTConfig(**config_args)
model = GPT(config)

sd = model.state_dict()
sd_keys = sd.keys()
sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

# init a huggingface/transformers model
model_hf = GPT2LMHeadModel.from_pretrained(model_type)
sd_hf = model_hf.state_dict()

# copy while ensuring all of the parameters are aligned and match in names and shapes
sd_keys_hf = sd_hf.keys()
sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
for k in sd_keys_hf:
	if any(k.endswith(w) for w in transposed):
		# special treatment for the Conv1D weights we need to transpose
		assert sd_hf[k].shape[::-1] == sd[k].shape
		with torch.no_grad():
			sd[k].copy_(sd_hf[k].t())
	else:
		# vanilla copy over the other parameters
		assert sd_hf[k].shape == sd[k].shape
		with torch.no_grad():
				sd[k].copy_(sd_hf[k])



torch.save(sd, "out/GPT2.pt")
