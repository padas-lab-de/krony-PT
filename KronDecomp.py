import torch
import torch.nn as nn

from einops import rearrange
from typing import Tuple

# this should've been already pre-installed. Update your python.

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

print(">>> Loading")
checkpoint = torch.load('out-shakespeare-char/ckpt.pt', map_location=device)
print(f"checkpoint keys: {[i for i in checkpoint.keys()]}")
print(">>> Loading \n")

checkpoint_model_args = checkpoint['model_args']
state_dict = checkpoint["model"] 

# contains the actual stuff. The real stuff.

unwanted_prefix = '_orig_mod.'

for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

list_state_dict = list(state_dict.items())

print("\n>>> the weights\n")


def print_params(list_state_dict):
    num_params = 0
    for (i,j) in list_state_dict:
        n = torch.numel(j)
        print(i, n)
        num_params += n
    print(f"\n Total # params: {num_params:_} \n")


# Kron Decompostion:

mlp0_c_fc = state_dict["transformer.h.0.mlp.c_fc.weight"]
mlp0_c_proj = state_dict["transformer.h.0.mlp.c_proj.weight"]

mlp0_c_fc1, mlp0_c_fc2 = kronecker_decompose(mlp0_c_fc, 1536, 32, k=2)
mlp0_c_proj1, mlp0_c_proj2 = kronecker_decompose(mlp0_c_proj, 32, 1536, k=2)

checkpoint["model"]["transformer.h.0.mlp0_c_fc1 "] = mlp0_c_fc1
checkpoint["model"]["transformer.h.0.mlp0_c_fc2"] = mlp0_c_fc2 
checkpoint["model"]["transformer.h.0.mlp0_c_proj1"] = mlp0_c_proj1 
checkpoint["model"]["transformer.h.0.mlp0_c_proj2"] = mlp0_c_proj2 


# saving the model

