import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from model_origin import *
from transformers import GPT2LMHeadModel

# this module has some utils functions, load the config, just to ease visual.
# TODO: this should be properly managed later.
from use import *

# model init using our configs.
config = GPTConfig(**config_args)
model = GPT(config)


checkpoint = torch.load("out/GPT2_3_11.pt")
model.load_state_dict(checkpoint)\

print("G shit only")



ctx =  torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
model.to(device)




print("Loss being computed!")
print(estimate_loss())

# Now I want to evaluate this model on wikitest-103.
# Now I want the perplexity.
# TODO: same for a distilGPT.



# different KP dimensions:









