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

    # Reshape and permute A, then perform SVD
    A = rearrange(A, "... (m m2) (n n2) -> ... (m n) (m2 n2)", m=m, m2=m2, n=n, n2=n2)
    u, s, v = torch.svd_lowrank(A, q=k, niter=niter)

    # Unflatten the factors
    u = rearrange(u, "... (m n) k -> ... k m n", m=m, n=n, k=k)
    v = rearrange(v, "... (m2 n2) k -> ... k m2 n2", m2=m2, n2=n2, k=k)

    scale = s[..., None, None].sqrt()
    return u * scale, v * scale



device = torch.device("cuda")

## GPU runtime innit:
print(">>> GPU burn")
x =  torch.randn(500,500, device = device)
_ = x@x
## GPU burned and happy.


# loading:   both should happen from terminal.
print(">>> Loading the ckpt and config")
# automate this shit, from terminal 

checkpoint_origin = torch.load('out-shakespeare-char/ckpt.pt', map_location=device)

# fix this prefix shit for once and all.
unwanted_prefix = '_orig_mod.'

state_dict = checkpoint_origin["model"]
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

#checkpoint_VL =   torch.load('out-shakespeare-char/ckpt_VL.pt', map_location=device)
print(">>> Done")


def kron_it(checkpoint, config: dict, fac: int):
    """
    conf should be written as follows:
	config = {"cfc":[n,m], "cproj":[n,m]}
    factors: number of factors

    TODO:
        * add checks / assert of dimension. 
        * have a manuel way of inputs from terminal, man be a professional
        * completely automate this shit.
    """

    # new checkpoint 
    checkpoint_VL = {i : checkpoint[i] for i in checkpoint if i!= "model"}
    checkpoint_VL["model"] = {}

    n_layer = checkpoint["model_args"]["n_layer"]
    nms_origin = set(checkpoint["model"].keys())

    for i in range(n_layer):
        c_fc_key = f"transformer.h.{i}.mlp.c_fc.weight"
        c_proj_key = f"transformer.h.{i}.mlp.c_proj.weight"

	# pop these two items cuz the y get copied eventually to the new ckpt
        # Perform kronecker decomposition and store the values in respective lists


        cfc_h = kronecker_decompose(
            checkpoint["model"][c_fc_key],
            config["c_fc"][0],
            config["c_fc"][1],
            k = fac
        )

        cproj_h = kronecker_decompose(
            checkpoint["model"][c_proj_key],
            config["c_proj"][0],
            config["c_proj"][1],
            k = fac
        )
        
        for f in range(fac):
            for k in range(2):
                fc = f"transformer.h.{i}.mlp.c_fc_{f}_{k}"
                proj = f"transformer.h.{i}.mlp.c_proj_{f}_{k}" 
                checkpoint_VL["model"][fc]   = cfc_h[k][f]
                checkpoint_VL["model"][proj] =  cproj_h[k][f] 

    for i in nms_origin:
	    if "c_fc" not in i and "c_proj" not in i:	
	            checkpoint_VL["model"][i] = checkpoint["model"][i]

    #custom_name = f"ckpt_{n}_{m}_{fac}"
    #torch.save(checkpoint_VL, "out/{custom_name}.pt")

    return checkpoint_VL["model"]

def param(checkpoint, pn, layer):
    """"
	pn is so far either {"fc", "proj"}
    """
    return checkpoint["model"][f"transformer.h.{layer}.mlp.c_{pn}.weight"]


conf = {"c_fc"   : [512,24], "c_proj" : [24,512]}

# the decompostion becomes almost useless with more rank.. hence.
# this could make a good small paragraph, Effectiveness of Van Loan
# conclusion, no benefits whatsoever compared to Random init.

new = kron_it(checkpoint_origin, conf, 4)

def c(new, pn, layer):
    h0 = [i for i in  list(new.keys()) if f"{layer}.mlp.c_{pn}" in i]
    return torch.stack([torch.kron(new[h0[i]], new[h0[i+1]])  for i in range(0, len(h0),2)]).sum(dim=0)


"""
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
