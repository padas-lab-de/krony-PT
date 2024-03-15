
sd = {}

lay = lambda layer : [i for i in sd.keys() if i.startswith(f"transformer.h.{layer}")]
lay_w = lambda layer : [i for i in sd.keys() if i.startswith(f"transformer.h.{layer}") and  any([i.endswith("_1"),i.endswith("_0")])]


# for the struct 96, 24
# Quickly instantiate a model:

from m2 import *
if True:
    config_args = dict(
        n_layer=12, 
        n_head=12, 
        n_embd=768,
        vocab_size = 50257,
        block_size = 1024,
        bias = True,
        dim_1 = 768,
        dim_2 = 768, 
        factors = 1
    )

krony_conf = KronyGPTConfig(**config_args)
krony = KronyGPT(krony_conf)
keys_all = list(krony.state_dict().keys())

p0 = "./OG-checks/1350.pt"
sd0 = torch.load(p0)

keys_rest = [i for i in keys_all if i not in sd0.keys()]

p1 = "./VLs/VL_94_24_1.pt"
sd1 = torch.load(p1)

for ky in keys_rest:
    x = ky[:-3] + ky[-2:]
    print(x)
    sd0[ky] = sd1[x]

keys = list(sd0.keys())
pt1 = [i for i in keys if any([i.endswith("proj2_0"), i.endswith("fc2_0")])]
pt2 = [i for i in keys if any([i.endswith("proj_0"), i.endswith("fc_0")])]

krony.load_state_dict(sd0)
sdxx = krony.state_dict()["transformer.h.0.mlp.c_fc2_0"]
sdyy = krony.state_dict()["transformer.h.0.mlp.c_fc_0"]
print(sdxx, sdyy)
for i in pt1:
    sd0[i] = 1/4 * sd0.pop(i)
    
for i in pt2:
    sd0[i] = 3/4 * sd0.pop(i)
    
    
krony.load_state_dict(sd0)
sdxx = krony.state_dict()["transformer.h.0.mlp.c_fc2_0"]
sdyy = krony.state_dict()["transformer.h.0.mlp.c_fc_0"]
print(sdxx, sdyy)