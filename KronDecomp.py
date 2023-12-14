import torch
import torch.nn as nn

from einops import rearrange
from typing import Tuple

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
checkpoint = torch.load('out-shakespeare-char/ckpt.pt', map_location=device)
print(f"checkpoint keys: {[i for i in checkpoint.keys()]}")
print(">>> Loading Done\n")

checkpoint_model_args = checkpoint['model_args']
state_dict = checkpoint["model"] 

# contains the actual stuff. The real stuff.

unwanted_prefix = '_orig_mod.'

for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

list_state_dict = list(state_dict.items())


def print_params(list_state_dict):
    num_params = 0
    for (i,j) in list_state_dict:
        n = torch.numel(j)
        print(i, n)
        num_params += n
    print(f"\n Total # params: {num_params:_} \n")




## Decomposition, and filling the checkpoint:

print(f"Kron. Decomposition 0")

h_mlp_c_fc_1_list, h_mlp_c_fc_2_list = [], []
h_mlp_c_proj_1_list, h_mlp_c_proj_2_list = [], []

for i in range(6):
    c_fc_key = f"transformer.h.{i}.mlp.c_fc.weight"
    c_proj_key = f"transformer.h.{i}.mlp.c_proj.weight"

    fc1 = f"transformer.h.{i}.mlp.c_fc_1"
    fc2 = f"transformer.h.{i}.mlp.c_fc_2"
    proj1 = f"transformer.h.{i}.mlp.c_proj_1"
    proj2 = f"transformer.h.{i}.mlp.c_proj_2"

    # Perform kronecker decomposition and store the values in respective lists
    h_mlp_c_fc_1, h_mlp_c_fc_2 = kronecker_decompose(checkpoint["model"][c_fc_key], 1536, 32)
    h_mlp_c_proj_1, h_mlp_c_proj_2 = kronecker_decompose(checkpoint["model"][c_proj_key], 32, 1536)

    checkpoint["model"][fc1]   = h_mlp_c_fc_1.squeeze(0) 
    checkpoint["model"][fc2]   =  h_mlp_c_fc_2.squeeze(0) 
    checkpoint["model"][proj1] =  h_mlp_c_proj_1.squeeze(0)
    checkpoint["model"][proj2] =   h_mlp_c_proj_1.squeeze(0)

print(f"Kron. Decomposition 1")




"""
## Some useful stuff for when you debug on .ipynb

from model import GPTConfig, KronyGPT 

args = checkpoint["model_args"]
conf =  GPTConfig(**args)
model = KronyGPT(conf)

model_state_dict =  model.state_dict() 
model_params_names = set(model_state_dict)

ckpt_new = {i : model_state_dict[i] 
            for i in model_state_dict 
            if ".mlp.c_fc.weight" not in i and ".mlp.c_proj.weight" not in i 
            }

ckpt2_names = set(ckpt_new.keys())

# number of params:
print(f"{sum(param2[i].numel() for i in params2):_}")


Configuring the new chekckpoint:

chpt2_list = list(checkpoint.keys())
chpt2_list.remove("model")

checkpoint2 = {}
for i in chpt2_list:
    checkpoint2[i] = checkpoint[i]


checkpoint2["model"] = ckpt_new


"""