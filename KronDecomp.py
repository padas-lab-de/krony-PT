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


device = torch.device("cuda:2")

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



"""

import torch 
import torch.nn as nn
from model import GPTConfig, GPT, KronyGPT

checkpoint = torch.load('out-shakespeare-char/ckpt.pt')
model_args = checkpoint["model_args"]

# create the model
gptconf = GPTConfig(**model_args)
model = KronyGPT(gptconf)

#state_dict = checkpoint['model']
#model.load_state_dict(state_dict)
"""


h0_mlp_c_fc_1, h0_mlp_c_fc_2  =     kronecker_decompose(checkpoint["model"]["transformer.h.0.mlp.c_fc.weight"]  , 1536, 32)
h0_mlp_c_proj_1, h0_mlp_c_proj_2 =  kronecker_decompose(checkpoint["model"]["transformer.h.0.mlp.c_proj.weight"], 32, 1536)

h1_mlp_c_fc_1, h1_mlp_c_fc_2  =     kronecker_decompose(checkpoint["model"]["transformer.h.1.mlp.c_fc.weight"]  , 1536, 32) 
h1_mlp_c_proj_1, h1_mlp_c_proj_2 =  kronecker_decompose(checkpoint["model"]["transformer.h.1.mlp.c_proj.weight"], 32, 1536)

h2_mlp_c_fc_1, h2_mlp_c_fc_2  =     kronecker_decompose(checkpoint["model"]["transformer.h.2.mlp.c_fc.weight"]  , 1536, 32) 
h2_mlp_c_proj_1, h2_mlp_c_proj_2 =  kronecker_decompose(checkpoint["model"]["transformer.h.2.mlp.c_proj.weight"], 32, 1536)

h3_mlp_c_fc_1, h3_mlp_c_fc_2  =     kronecker_decompose(checkpoint["model"]["transformer.h.3.mlp.c_fc.weight"]  , 1536, 32) 
h3_mlp_c_proj_1, h3_mlp_c_proj_2 =  kronecker_decompose(checkpoint["model"]["transformer.h.3.mlp.c_proj.weight"], 32, 1536)

h4_mlp_c_fc_1, h4_mlp_c_fc_2  =     kronecker_decompose(checkpoint["model"]["transformer.h.4.mlp.c_fc.weight"]  , 1536, 32) 
h4_mlp_c_proj_1, h4_mlp_c_proj_2 =  kronecker_decompose(checkpoint["model"]["transformer.h.4.mlp.c_proj.weight"], 32, 1536)

h5_mlp_c_fc_1, h5_mlp_c_fc_2  =     kronecker_decompose(checkpoint["model"]["transformer.h.5.mlp.c_fc.weight"]  , 1536, 32) 
h5_mlp_c_proj_1, h5_mlp_c_proj_2 =  kronecker_decompose(checkpoint["model"]["transformer.h.5.mlp.c_proj.weight"], 32, 1536)

checkpoint["model"]["transformer.h.0.mlp.c_fc_1"] =   h0_mlp_c_fc_1.squeeze(0)
checkpoint["model"]["transformer.h.0.mlp.c_fc_2"] =   h0_mlp_c_fc_2.squeeze(0)
checkpoint["model"]["transformer.h.0.mlp.c_proj_1"] = h0_mlp_c_proj_1.squeeze(0)
checkpoint["model"]["transformer.h.0.mlp.c_proj_2"] = h0_mlp_c_proj_2.squeeze(0)

checkpoint["model"]["transformer.h.1.mlp.c_fc_1"] =  h1_mlp_c_fc_1.squeeze(0)
checkpoint["model"]["transformer.h.1.mlp.c_fc_2"] =  h1_mlp_c_fc_2.squeeze(0)
checkpoint["model"]["transformer.h.1.mlp.c_proj_1"]= h1_mlp_c_proj_1.squeeze(0)
checkpoint["model"]["transformer.h.1.mlp.c_proj_2"]= h1_mlp_c_proj_2.squeeze(0)
          
checkpoint["model"]["transformer.h.2.mlp.c_fc_1"] =    h2_mlp_c_fc_1.squeeze(0)
checkpoint["model"]["transformer.h.2.mlp.c_fc_2"] =    h2_mlp_c_fc_2.squeeze(0)
checkpoint["model"]["transformer.h.2.mlp.c_proj_1"] =  h2_mlp_c_proj_1.squeeze(0)
checkpoint["model"]["transformer.h.2.mlp.c_proj_2"] =  h2_mlp_c_proj_2.squeeze(0)
          
checkpoint["model"]["transformer.h.3.mlp.c_fc_1"] =   h3_mlp_c_fc_1.squeeze(0)
checkpoint["model"]["transformer.h.3.mlp.c_fc_2"] =   h3_mlp_c_fc_2.squeeze(0)
checkpoint["model"]["transformer.h.3.mlp.c_proj_1"] =  h3_mlp_c_proj_1.squeeze(0)
checkpoint["model"]["transformer.h.3.mlp.c_proj_2"] =  h3_mlp_c_proj_2.squeeze(0)

checkpoint["model"]["transformer.h.4.mlp.c_fc_1"] =  h4_mlp_c_fc_1.squeeze(0)
checkpoint["model"]["transformer.h.4.mlp.c_fc_2"] =  h4_mlp_c_fc_2.squeeze(0)
checkpoint["model"]["transformer.h.4.mlp.c_proj_1"] = h4_mlp_c_proj_1.squeeze(0)
checkpoint["model"]["transformer.h.4.mlp.c_proj_2"] = h4_mlp_c_proj_2.squeeze(0)

checkpoint["model"]["transformer.h.5.mlp.c_fc_1"] =   h5_mlp_c_fc_1.squeeze(0)
checkpoint["model"]["transformer.h.5.mlp.c_fc_2"] =   h5_mlp_c_fc_2.squeeze(0)
checkpoint["model"]["transformer.h.5.mlp.c_proj_1"] =  h5_mlp_c_proj_1.squeeze(0)
checkpoint["model"]["transformer.h.5.mlp.c_proj_2"] =  h5_mlp_c_proj_2.squeeze(0)

"""
w0 = checkpoint["model"]["transformer.h.0.mlp.c_fc.weight"] 
w01, wo2  =   kronecker_decompose( w0 , 1536, 32)

w1 = checkpoint["model"]["transformer.h.0.mlp.c_proj.weight"]
w11, w12  =   kronecker_decompose( w0 , 1536, 32)
"""


# a useful code to detect the old params from checkoint["model"]


"""
for i in checkpoint["model"]:
    if "mlp.c_fc.weight" in i or "mlp.c_proj.weight" in i:
        # pop it like you mean it!

# one of my most proud one liners

params2 = {i:checkpoint["model"][i] 
                for i in checkpoint["model"] 
                    if "mlp.c_fc.weight" not  in i 
                    and  
                    "mlp.c_proj.weight" not  in i
        }        


"""

"""
for a quick demos:



args = checkpoint["model_args"]
conf =  GPTConfig(args)
model = KronyGPT(cond)


# number of params:
print(f"{sum(param2[i].numel() for i in params2):_}")

model.state_dict()

"""