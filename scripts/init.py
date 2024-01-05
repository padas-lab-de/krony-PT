import torch

device = "cuda"

print(">> loading data")
sd = torch.load('../out/GPT2.pt', map_location=device)

nms_origin = list(sd.keys())

def init(sd, n_layer: int):
	new = dict()
	for i in range(n_layer):
		print(f"Processing layer {i}")

		c_fc_key = f"transformer.h.{i}.mlp.c_fc.weight"
		c_proj_key = f"transformer.h.{i}.mlp.c_proj.weight"

		fc = sd[c_fc_key]
		proj = sd[c_proj_key]


		cfc_h = fc.view(fc.shape[0], fc.shape[1]//2, 2)[:,:,1]
		cproj_h = proj.view(proj.shape[0]//2, 2, proj.shape[1])[:,1,:]

		# cleaning the original checkpoint.
		nms_origin.remove(c_fc_key)
		nms_origin.remove(c_proj_key)
		nms_origin.remove(f"{c_fc_key[:-6]}bias")
		nms_origin.remove(f"{c_proj_key[:-6]}bias")

		for k in range(2):
			fc = f"transformer.h.{i}.mlp.c_fc_{0}_{k}"
			proj = f"transformer.h.{i}.mlp.c_proj_{0}_{k}" 
			if k == 0:
				new[fc]   = cfc_h
				new[proj] = cproj_h
			else:
				new[fc]   =  torch.tensor([0,1]).to(device)
				new[proj] =  torch.tensor([[0],[1]]).to(device)
	return new

new = init(sd, 12)
nms = list(new.keys())

for w in nms_origin:
	if w not in nms:
		new[w] = sd[w]

print("saving!")
torch.save(new, "../out/GPT2_prune_init.pt")


"""
# some testing cuz why not
def showme(i)  :
	cfc = f"transformer.h.{i}.mlp.c_fc.weight"
	cproj = f"transformer.h.{i}.mlp.c_proj.weight"

	fc = sd[cfc]
	proj = sd[cproj]

	x0 = new[f"transformer.h.{i}.mlp.c_fc_{0}_{0}"]
	x1 = new[f"transformer.h.{i}.mlp.c_fc_{0}_{1}"]
	x2=  new[f"transformer.h.{i}.mlp.c_proj_{0}_{0}"]
	x3 = new[f"transformer.h.{i}.mlp.c_proj_{0}_{1}"]

	y1 = torch.kron(x0, x1)
	y2 = torch.kron(x2, x3)

	print(torch.sum(y1==fc), torch.sum(y2==proj))


for i in range(12):
	showme(i)

"""