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

print(">>> GPU init")
x =  torch.randn(10,10, device = device)
_ = x@x
print(">>> init 2")
_ = x@x

print(">>> Loading the ckpt")

sd = torch.load('out/GPT2_3_11.pt', map_location=device)
nms_origin = list(sd.keys())

def kron_it(checkpoint, config: dict, fac: int):
	n_layer = 12      
	checkpoint_VL = dict()
	checkpoint_VL["model"] = dict()

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

		for f in range(fac):
			for k in range(2):
				fc = f"transformer.h.{i}.mlp.c_fc_{f}_{k}"
				proj = f"transformer.h.{i}.mlp.c_proj_{f}_{k}" 
				checkpoint_VL["model"][fc]   = cfc_h[k][f]
				checkpoint_VL["model"][proj] =  cproj_h[k][f] 

		nms = list(checkpoint_VL["model"].keys())

		for w in nms_origin:
			if w not in nms:
				checkpoint_VL["model"][w] = checkpoint[w]

		#custom_name = f"ckpt_{n}_{m}_{fac}"
		#torch.save(checkpoint_VL, "out/{custom_name}.pt")

	return checkpoint_VL["model"]



conf = {"fc"   : (3072,384), "proj" : (384, 3072)}
sd_VL1 = kron_it(sd, conf, 1)

torch.save(sd_VL1, "out/GPT2_VL1.pt")

# the decompostion becomes almost useless with more rank.. hence.
# this could make a good small paragraph, Effectiveness of Van Loan
# conclusion, no benefits whatsoever compared to Random init.

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
