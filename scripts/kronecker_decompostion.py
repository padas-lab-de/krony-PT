"""
This script:
1. Loads the GPT2 checkpoints from HF (that are already stored localy)
2. Decomposes the weights using the Van Loan method.
3. Creates a new state dict that matches the KronyGPT state dict
4. Saves the checkpoint in out/


To play with it:
change the **conf** dict to your liking.

"""

import torch
from einops import rearrange
from mmodel import *
from model_origin import *

# add original link to the implementation. > I'm following her on Twitter, Hailey smth.
def kronecker_decompose(A , m: int, n: int, *, k: int = 1, niter: int = 10):
	"""
		Frobenius-optimal decomposition of `A` into a sum of `k` Kronecker products.
		Algorithm from Van Loan and Pitsianis (1993),
		"Approximation with Kronecker Products"
		<https://bit.ly/46hT5aY>.

			Args:
		A: Matrix or batch of matrices to decompose, of shape (..., m * m2, n * n2)
		m: Desired number of rows in the left Kronecker factor(s)
		n: Desired number of columns in the left Kronecker factor(s)
		k: Number of Kronecker factors
		niter: Number of iterations for the low rank SVD algorithm

		Returns:
		Tuple of Kronecker factors (`left`, `right`) of shape `(..., k, m, n)` and
		`(..., k, A.shape[-2] // m, A.shape[-1] // n)` respectively.

		Raises:
		AssertionError: If the dimensions of `A` are not compatible with the desired
		number of rows and columns in the left Kronecker factor.

	"""

	m2, n2 = A.shape[-2] // m, A.shape[-1] // n
	assert A.shape[-2:] == (m * m2, n * n2), "Dimensions do not match"

	A = rearrange(A, "... (m m2) (n n2) -> ... (m n) (m2 n2)", m=m, m2=m2, n=n, n2=n2)
	u, s, v = torch.svd_lowrank(A, q=k, niter=niter)

	u = rearrange(u, "... (m n) k -> ... k m n", m=m, n=n, k=k)
	v = rearrange(v, "... (m2 n2) k -> ... k m2 n2", m2=m2, n2=n2, k=k)

	scale = s[..., None, None].sqrt()
	return u * scale, v * scale

device = torch.device("cuda")

from transformers import GPT2LMHeadModel

gpt = GPT2LMHeadModel.from_pretrained("gpt2").state_dict()
k_origin= list(gpt.keys())

def kron_it(checkpoint, config: dict):
	print(f"This will return two factors of dims {config['dim']} \
     and {(3072//config['dim'][0], 768//config['dim'][1])}")

	n_layer = 12      
	fac = config["n_factors"]
	new = dict()

	for i in range(n_layer):
		c_fc_key = f"transformer.h.{i}.mlp.c_fc.weight"
		c_proj_key = f"transformer.h.{i}.mlp.c_proj.weight"

		cfc_h = kronecker_decompose(
			checkpoint[c_fc_key],
			config["dim"][0],
			config["dim"][1],
			k = fac
			)

		cproj_h = kronecker_decompose(
			checkpoint[c_proj_key],
			config["dim"][1],
			config["dim"][0],
			k = fac
			)

		for f in range(fac):
			for k in range(2):
				fc = f"transformer.h.{i}.mlp.c_fc_{f}_{k}"
				proj = f"transformer.h.{i}.mlp.c_proj_{f}_{k}" 
				new[fc]   = cfc_h[k][f]
				new[proj] =  cproj_h[k][f] 

		new[f"{c_fc_key[:-7]}_bias"]   = gpt[f"{c_fc_key[:-7]}.bias"]
		new[f"{c_proj_key[:-7]}_bias"] = gpt[f"{c_proj_key[:-7]}.bias"]
		
	return new

def kron_it_2(checkpoint, config: dict):
	print(f"This will return two factors of dims {config['dim']} \
     and {(3072//config['dim'][0], 768//config['dim'][1])}")

	n_layer = 12      
	fac = config["n_factors"]
	new = dict()

	for i in range(n_layer):
		c_fc_key = f"transformer.h.{i}.mlp.c_fc.weight"
		c_proj_key = f"transformer.h.{i}.mlp.c_proj.weight"

		cfc_h = kronecker_decompose(
			checkpoint[c_fc_key],
			config["dim"][0],
			config["dim"][1],
			k = fac
			)

		cproj_h = kronecker_decompose(
			checkpoint[c_proj_key],
			config["dim"][1],
			config["dim"][0],
			k = fac
			)

		for k in range(2):
			fc = f"transformer.h.{i}.mlp.c_fc_{k}"
			proj = f"transformer.h.{i}.mlp.c_proj_{k}" 
			new[fc]   = cfc_h[k]
			new[proj] =  cproj_h[k]

		new[f"{c_fc_key[:-7]}_bias"]   = gpt[f"{c_fc_key[:-7]}.bias"]
		new[f"{c_proj_key[:-7]}_bias"] = gpt[f"{c_proj_key[:-7]}.bias"]
		
	return new
# change here 

dim1 = 3
dim2 = 12
facss = 2
conf = {"dim"   : (dim1, dim2),  # the dims of A (m_1, n_1) following latex notation
		 "n_factors" : facss
}

# Setting up a quick KronyGPT

config_args = dict(
	n_layer=12, 
	n_head=12, 
	n_embd=768,
	vocab_size = 50257,
	block_size = 1024,
	bias = True,
	dim_1 = dim1,
	dim_2 = dim2,
	factors = facss 
)

krony_conf = KronyGPTConfig(**config_args)
kronyG = KronyGPT(krony_conf)
krony_sd   = kronyG.state_dict()
k_krony    = krony_sd.keys()

sd = kron_it_2(gpt, conf)

transposed = ['attn.c_attn.weight', 'attn.c_proj.weight']
#transposed = ['attn.c_attn.weight']

for k in k_origin:
	if any(k.endswith(w) for w in transposed):
		sd[k] = gpt[k].t()

rest = [i for i in k_krony if i not in sd.keys()]

for i in rest:
    sd[i] = gpt[i]

kronyG.load_state_dict(sd) 

#print("3. Saving!")
#torch.save(new, "out2/VL_768_768.pt")


##### Testing: ##########################################################################

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

	import numpy as np

	batch_size = 12
	block_size = 1024
	device = "cuda"
	##
	path = 'data/openwebtext/'
	train_data = np.memmap(f'{path}train.bin', dtype=np.uint16, mode='r')
	val_data = np.memmap(f'{path}val.bin', dtype=np.uint16, mode='r')
	def get_batch(split):
		data = train_data if split == 'train' else val_data
		ix = torch.randint(len(data) - block_size, (batch_size,))
		x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
		y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
		return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

x, y = get_batch("train")

gpt0 = GPT(conf)
gpt0 = gpt0.from_pretrained("gpt2")
sd0  = gpt0.state_dict()
gpt0.to(device)
kronyG.to(device)


r0 = gpt0(x)
r2 = kronyG(x)

for i in range(12):
    print("Batch > \n")
    print(f"{r0[0][i][0,:10]}")
    print(f"{r2[0][i][0,:10]}")




"""
nms = list(sd.keys())
one_block = nms[2:14]

k, kk = nms[-1], sd[nms[-1]].numel()
print(f"Snitch > {k}")

total = sum(j.numel() for i,j in sd.items()) - kk
print(f"Total number of ds:  {total:_}")

print(f"details of one block:")

b1 = 0
inms = []
for i in nms:
	if "mlp.c_proj.weight"  in i or "mlp.c_fc.weight" in i:
		inms.append(i)
		_ = sd[i].numel()
		print(i, _)
		b1 += _

print(f"for one block we got {b1:_}")

def param(checkpoint, pn, layer):
# pn is so far either {"fc", "proj"}
		return checkpoint["model"][f"transformer.h.{layer}.mlp.c_{pn}.weight"]

		To generate a checkpoint:

		write a config:

		config = { 
param_1 : [dim1, dim2],
		  ...
			  param_N: [dim1, dim2]
		}

pass it to kronDecompose python3 kronDecompose.py --origin="ckpt1.pt" --config=config_file
this will automatically generate a checkpoint for you. withe the tag ckpt_p
"""
