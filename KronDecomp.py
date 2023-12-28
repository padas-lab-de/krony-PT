import torch
import torch.nn as nn

from einops import rearrange
from typing import Tuple

from model import GPT, GPTConfig
# add original link to the implementation. > twitter lol

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
x =  torch.randn(500,500, device = device)
_ = x@x
## GPU burned and happy.

print(">>> Loading")


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







def kron_it(checkpoint, n: int, m: int, fac: int):
    """
    n: first dim
    m: second dim
    fac: number of factors
    TODO:
        * add checks / assert of dimension. 
        * have a manuel way of inputs from terminal, man be a professional
        * completely automate this shit.
    """
    # new checkpoint 
    checkpoint_VL = {i : checkpoint[i] for i in checkpoint if i!= "model"}
    checkpoint_VL["model"] = {}

    config = checkpoint["config"]
    nms_origin = set(checkpoint["model"].keys())
    n_layer = config["_layer"]

    for i in range(n_layer):
        c_fc_key = f"transformer.h.{i}.mlp.c_fc.weight"
        c_proj_key = f"transformer.h.{i}.mlp.c_proj.weight"


        # Perform kronecker decomposition and store the values in respective lists

        cfc_h = kronecker_decompose(
            checkpoint_origin["model"][c_fc_key],
            n,
            m
        )

        cproj_h = kronecker_decompose(
            checkpoint_origin["model"][c_proj_key],
            n,
            m
        )
        
        for f in range(fac):
            for k in range(2):
                fc = f"transformer.h.{i}.mlp.c_fc_{f}{k}"
                proj = f"transformer.h.{i}.mlp.c_proj_{f}{k}"
                checkpoint_VL["model"][fc]   = cfc_h[k].squeeze(0) # cfc[] cproj[] needs to change below 
                checkpoint_VL["model"][proj] =  cproj_h[k].squeeze(0)  # cfc[] cproj[] needs to change below 

    for i in nms_origin:
            checkpoint_VL["model"][i] = checkpoint_origin["model"][i]

    custom_name = f"ckpt_{n}_{m}_{fac}"
    torch.save(checkpoint_VL, "out/{custom_name}.pt")
    return 1 



#print(set(model.state_dict().keys())== set(checkpoint_VL["model"].keys()))


