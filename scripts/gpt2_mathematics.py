import torch
from transformers import GPT2LMHeadModel
import numpy as np

## importing normal gpt2.
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
sd = gpt2.state_dict()
sd.pop("lm_head.weight")  # this is basically a duplicate of the embedding matrix

l_keys  = list(sd.keys())
tot = 0
tot = sum(p.numel() for pn,p in sd.items())
print("tot = ",tot)

k1 = [i for i in l_keys if     i.startswith("transformer.h.")]

k11 = [i for i in k1 if     any([i.endswith("mlp.c_fc.weight"), i.endswith("mlp.c_proj.weight")])]
k12 = [i for i in k1 if i not in k11  ]
k2 = [i for i in l_keys if not i.startswith("transformer.h.")]

assert set(l_keys) == set(k1+k2), "the fuck is going on?"



n = 3072
m = 768

n_pot = [i for i in range(1, int(np.sqrt(n))+1) if n % i == 0]
m_pot = [i for i in range(1, int(np.sqrt(m))+1) if m % i == 0]

comb = []
sums = set()
for n1 in n_pot:
	for m1 in m_pot:
		n2 = n // n1
		m2 = m // m1
		l1 = min(n1,m1)
		l2 = min(n2, m2)
		if l1*l2 == 768:
			s = n1*m1 + n2*m2
			sums.add(s)
			comb.append([s, (n1, m1), (n2, m2)])

comb = sorted(comb)



nms   = lambda layer, keys : [i for i in keys if i.startswith(f"transformer.h.{layer}.")]
n_pms = lambda l_keys , state_d :  sum(state_d[pn].numel() for pn in l_keys)

lay1 = nms(1,k1)


s2 = sum(sd[i].numel() for i in k2)
s11 = sum(sd[i].numel() for i in k11)
s12 = sum(sd[i].numel() for i in k12)

print(f"{s2+s11+s12:_}")

