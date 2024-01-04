import torch

device = "cuda"
sd = torch.load('../out/GPT2.pt', map_location=device)
nms_origin = list(sd.keys())

def init(sd, config: dict):
	n_layer = 12      
	fac = config["n_factors"]
	new = dict()

	for i in range(n_layer):
		c_fc_key = f"transformer.h.{i}.mlp.c_fc.weight"
		c_proj_key = f"transformer.h.{i}.mlp.c_proj.weight"

		fc = sd[c_fc_key]
		proj = sd[c_proj_key]

		cfc_h = fc.view(fc.shape[0]//2, 2, fc.shape[1])
		cproj_h = proj.view(proj.shape[0]//2, 2, proj.shape[1])
        
        # argmax of abs:
		cfc_h_argx_abs = torch.argmax(torch.abs(cfc_h), dim=1)
		cproj_h_argx_abs = torch.argmax(torch.abs(cproj_h), dim=1)

        # selecting
		cfc_h   = torch.gather(cfc_h , 1, cfc_h_argx_abs.unsqueeze(1)) 
		cproj_h = torch.gather(cproj_h , 1, cproj_h_argx_abs.unsqueeze(1))

		# cleaning the original checkpoint.
		nms_origin.remove(c_fc_key)
		nms_origin.remove(c_proj_key)
		nms_origin.remove(f"{c_fc_key[:-6]}bias")
		nms_origin.remove(f"{c_proj_key[:-6]}bias")

	for k in range(2):
		fc = f"transformer.h.{i}.mlp.c_fc_{0}_{k}"
		proj = f"transformer.h.{i}.mlp.c_proj_{0}_{k}" 
		if k == 0:
			new[fc]   = cfc_h[k][f]
			new[proj] =  cproj_h[k][f] 
		else:
			new[fc]   = cfc_h[k][f]
			new[proj] =  cproj_h[k][f] 
	return new

# change here 

conf = {"fc"   : (3072,384), 
		"proj" : (384, 3072),
		"n_factors" : 1
}

new = init(conf)


