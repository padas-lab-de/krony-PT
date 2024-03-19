import torch
from transformers import GPT2LMHeadModel
from model import *

device = "cuda"

print("Loading Krony")
if True: # Setting up a quick KronyGPT
	config_args = dict(
		n_layer=12, 
		n_head=12, 
		n_embd=768,
		vocab_size = 50257,
		block_size = 1024,
		bias = True,
		dim_1 = 384,
		dim_2 = 3072,
		factors = 1 
	)

print("Loading KronyGPT")
krony_conf = KronyGPTConfig(**config_args)
kronyG = KronyGPT(krony_conf)
krony_sd   = kronyG.state_dict()
k_krony    = krony_sd.keys()

print("Loading GPT2")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
sd2	 = gpt2.state_dict()
nms	 = list(sd2.keys())

new_sd = {}  # the sd we gon fill to load eventually on Krony

common = [pn for pn in nms if pn in k_krony]
transposed = ['attn.c_attn.weight', 'attn.c_proj.weight']
common_t	= [pn for pn in common if any(pn.endswith(tr) for tr in transposed)]
common_not_t= [pn for pn in common if not any(pn.endswith(tr) for tr in transposed)]

for pn in common_t:
	new_sd[pn] = sd2[pn].t()

for pn in common_not_t:
	new_sd[pn] = sd2[pn]

rest = [pn for pn in krony_sd if pn not in list(new_sd.keys())]
# we are left to fill the mlps: 
# 1. Kronecker Factors > need work
# 2. Biases > Easy
rest_bias = [pn for pn in rest if pn.endswith("bias")]
rest_kron = [pn for pn in rest if not pn.endswith("bias")]

for pn in rest_bias:
	new_sd[pn] = sd2[f"{pn[:-5]}.bias"]
 
rest2 = [pn for pn in krony_sd if pn not in list(new_sd.keys())] # just making sure for debugging purposes
 
 
xx = "transformer.h.0.mlp.c_fc.weight"
x1 = "transformer.h.0.mlp.c_fc_0"
x2 = "transformer.h.0.mlp.c_fc_1"

# keep in mind:
# fc_0 is 1 384 3072
# proj_0  1 3072 384

for i in range(2) :
	print(f"For layer {i} >>")
	c_fc_key = f"transformer.h.{i}.mlp.c_fc.weight"
	c_proj_key = f"transformer.h.{i}.mlp.c_proj.weight"

	fc 	 = sd2[c_fc_key]
	proj = sd2[c_proj_key]

	#fc1  = fc[0::2].unsqueeze(0)
	#proj1= proj[:, 0::2].unsqueeze(0)
 
	fc1  = fc[0::2]
	proj1= proj[:, 0::2]
 
	fc1   =  torch.tensor([[0,1]])
	proj1 =  torch.tensor([[0],[1]])

	print(fc.shape, proj.shape)
	print(fc1.shape, proj1.shape)
   
"""
	c_fc_key = f"transformer.h.{i}.mlp.c_fc.weight"
	c_proj_key = f"transformer.h.{i}.mlp.c_proj.weight"

	fc = sd[c_fc_key]
	proj = sd[c_proj_key]


	cfc_h = fc.view(fc.shape[0], fc.shape[1]//2, 2)[:,:,1]
	cproj_h = proj.view(proj.shape[0]//2, 2, proj.shape[1])[:,1,:]

	# cleaning the original checkpoint.
	nms_origin.remove(c_fc_key)
	nms_origin.remove(c_proj_key)
	nms_origin.remove(f"{c_fc_key[:-6]}bias")
	nms_origin.remove(f"{c_proj_key[:-6]}bias")

	for k in range(2):
		fc = f"transformer.h.{i}.mlp.c_fc_{0}_{k}"
		proj = f"transformer.h.{i}.mlp.c_proj_{0}_{k}" 
		if k == 0:
			new[fc]   = cfc_h
			new[proj] = cproj_h
		else:
			new[fc]   =  torch.tensor([[0,1]]).to(device)
			new[proj] =  torch.tensor([[0],[1]]).to(device)
return new


"""
# hey fuck off
"""
From GPT2 to Krony, you should handle 3 situtaitons:
1. Common parameters transposed
2. Common parameters NOT transposed
3. Kronecker Factors




def init(sd, n_layer: int):
	new = dict()
	for i in range(n_layer):
		print(f"Processing layer {i}")

		c_fc_key = f"transformer.h.{i}.mlp.c_fc.weight"
		c_proj_key = f"transformer.h.{i}.mlp.c_proj.weight"

		fc = sd[c_fc_key]
		proj = sd[c_proj_key]


		cfc_h = fc.view(fc.shape[0], fc.shape[1]//2, 2)[:,:,1]
		cproj_h = proj.view(proj.shape[0]//2, 2, proj.shape[1])[:,1,:]

		# cleaning the original checkpoint.
		nms_origin.remove(c_fc_key)
		nms_origin.remove(c_proj_key)
		nms_origin.remove(f"{c_fc_key[:-6]}bias")
		nms_origin.remove(f"{c_proj_key[:-6]}bias")

		for k in range(2):
			fc = f"transformer.h.{i}.mlp.c_fc_{0}_{k}"
			proj = f"transformer.h.{i}.mlp.c_proj_{0}_{k}" 
			if k == 0:
				new[fc]   = cfc_h
				new[proj] = cproj_h
			else:
				new[fc]   =  torch.tensor([[0,1]]).to(device)
				new[proj] =  torch.tensor([[0],[1]]).to(device)
	return new

"""


"""

print("saving!")
torch.save(new, "../out/GPT2_prune_init.pt")

for i in range(n_layer):
	print(f"Processing layer {i}")

	c_fc_key = f"transformer.h.{i}.mlp.c_fc.weight"
	c_proj_key = f"transformer.h.{i}.mlp.c_proj.weight"

	# cleaning the original checkpoint.

	nms_origin.remove(c_fc_key)
	nms_origin.remove(c_proj_key)
	nms_origin.remove(f"{c_fc_key[:-6]}bias")
	nms_origin.remove(f"{c_proj_key[:-6]}bias")

	for k in range(2):
		fc = f"transformer.h.{i}.mlp.c_fc_{0}_{k}"
		proj = f"transformer.h.{i}.mlp.c_proj_{0}_{k}" 
		if k == 0:
			new[fc]   =  torch.normal(0, 0.02, size=(3072,384))
			new[proj] =  torch.normal(0, 0.02, size=(384,3072))
		else:
			new[fc]   =   torch.normal(0, 0.02, size=(1,2))  
			new[proj] =   torch.normal(0, 0.02, size=(2,1))

print("saving!")
torch.save(new, "../out/GPT2_rand_KP_init.pt")



# some testing cuz why not

det showme(i)  :
	cfc = f"transformer.h.{i}.mlp.c_fc.weight"
	cproj = f"transformer.h.{i}.mlp.c_proj.weight"

	fc = sd[cfc]
	proj = sd[cproj]

	x0 = new[f"transformer.h.{i}.mlp.c_fc_{0}_{0}"]
	x1 = new[f"transformer.h.{i}.mlp.c_fc_{0}_{1}"]
	x2=  new[f"transformer.h.{i}.mlp.c_proj_{0}_{0}"]
	x3 = new[f"transformer.h.{i}.mlp.c_proj_{0}_{1}"]

	y1 = torch.kron(x0, x1)
	y2 = torch.kron(x2, x3)

	print(torch.sum(y1==fc), torch.sum(y2==proj))


for i in range(12):
	showme(i)

"""