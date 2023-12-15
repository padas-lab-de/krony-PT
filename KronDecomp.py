import torch
import torch.nn as nn

from einops import rearrange
from typing import Tuple

from model import GPT, GPTConfig
# add original link to the implementation.

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


device = torch.device("cuda:3")

## GPU runtime innit:
x =  torch.randn(500,500, device = device)
_ = x@x
## GPU burned and happy.

print(">>> Loading")
checkpoint_origin = torch.load('out-shakespeare-char/ckpt.pt', map_location=device)

unwanted_prefix = '_orig_mod.'

state_dict = checkpoint_origin["model"]
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

#checkpoint_VL =   torch.load('out-shakespeare-char/ckpt_VL.pt', map_location=device)
print(">>> Done")


#Initial KronP decom. could be obtained as follows:

checkpoint_VL = {i : checkpoint_origin[i] for i in checkpoint_origin if i!= "model"}
checkpoint_VL["model"] = {}


print(f"Kron. Decomposition 0")

for i in range(6):
    c_fc_key = f"transformer.h.{i}.mlp.c_fc.weight"
    c_proj_key = f"transformer.h.{i}.mlp.c_proj.weight"

    fc1 = f"transformer.h.{i}.mlp.c_fc_1"
    fc2 = f"transformer.h.{i}.mlp.c_fc_2"
    proj1 = f"transformer.h.{i}.mlp.c_proj_1"
    proj2 = f"transformer.h.{i}.mlp.c_proj_2"

    # Perform kronecker decomposition and store the values in respective lists

    h_mlp_c_fc_1, h_mlp_c_fc_2 = kronecker_decompose(
        checkpoint_origin["model"][c_fc_key],
        1536,
        32
    )

    h_mlp_c_proj_1, h_mlp_c_proj_2 = kronecker_decompose(
        checkpoint_origin["model"][c_proj_key],
        32, 
        1536
    )

    checkpoint_VL["model"][fc1]   = h_mlp_c_fc_1.squeeze(0) 
    checkpoint_VL["model"][fc2]   =  h_mlp_c_fc_2.squeeze(0) 
    checkpoint_VL["model"][proj1] =  h_mlp_c_proj_1.squeeze(0)
    checkpoint_VL["model"][proj2] =   h_mlp_c_proj_2.squeeze(0)

print(f">>> Done")


model_args = checkpoint_origin["model_args"]
conf = GPTConfig(**model_args)
model = GPT(conf)

for i in model.state_dict():
    if i in checkpoint_origin["model"].keys():
        #print(i)
        checkpoint_VL["model"][i] = checkpoint_origin["model"][i]

print(set(model.state_dict().keys())== set(checkpoint_VL["model"].keys()))

torch.save(checkpoint_VL, "out-shakespeare-char/ckpt_VL.pt")
