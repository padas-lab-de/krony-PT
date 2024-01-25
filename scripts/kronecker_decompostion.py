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
from typing import Tuple

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

# loadign ckpt
print("1. Loading GPT2 3.11 loss")

sd = torch.load('out/GPT2.pt', map_location=device)
nms_origin = list(sd.keys())

def kron_it(checkpoint, config: dict):
	n_layer = 12      
	fac = config["n_factors"]
	new = dict()

	for i in range(n_layer):
		c_fc_key = f"transformer.h.{i}.mlp.c_fc.weight"
		c_proj_key = f"transformer.h.{i}.mlp.c_proj.weight"

		cfc_h = kronecker_decompose(
			checkpoint[c_fc_key],
			config["fc"][0],
			config["fc"][1],
			k = fac
			)

		cproj_h = kronecker_decompose(
			checkpoint[c_proj_key],
			config["proj"][0],
			config["proj"][1],
			k = fac
			)

		# cleaning the original checkpoint.
		nms_origin.remove(c_fc_key)
		nms_origin.remove(c_proj_key)

		for f in range(fac):
			for k in range(2):
				fc = f"transformer.h.{i}.mlp.c_fc_{f}_{k}"
				proj = f"transformer.h.{i}.mlp.c_proj_{f}_{k}" 
				new[fc]   = cfc_h[k][f]
				new[proj] =  cproj_h[k][f] 
	return new




# change here 
conf = {"fc"   : (3072,384), 
		"proj" : (384, 3072),
		"n_factors" : 1
}

print("2. Decomposing")

sd1 = kron_it(sd, conf)
nms =  list(sd1.keys())

for w in nms_origin:
	if w not in nms:
		sd1[w] = sd[w]


nms2 =  list(sd1.keys())

#print("3. Saving!")
#torch.save(sd_VL_2_factors, "out/GPT2_VL_2_factors.pt")








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
