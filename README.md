This is a a detached fork of [NanoGPT @ Karpathy](https://github.com/karpathy/nanoGPT/) :goat:.

The goal is to factorize single weight matrices into a product of Kronecker Matrices. 

---
### Progress

Status: Initial code (seems like It) is working.

Some early reports on val loss:
* Link to wandb.

**Some prelims findings:**

* Original nanoModel is 10.7M, Facftorized model in 4.3M.
* KD is slow? (not really), but  robust to overfitting.
	* Q: Is it very robust or do the gradients get saturated? hence, no updates? 

---
**TODO:**

* Write code for distilled initialization of KronyMLP [IN-PROGRESS]
	* Test improved KP computations.

* Optimize compute 
	* Still haven't used any KP props. But I sus. that it would help.

* Experiment more / write results.
	* different n / m for KroneckerDecom
	* Try the kronecker decompostion using the lib.

* MoE? what is it? can we decompose it?

**DONE**

* Report using wandb, log everything. [DONE]

* Fix the resume "training"
	* fix the iter_num in .pt files. [DONE]
	* I only want to pick the weights (state_dict()) everything else should be same as Training from scratchi [DONE]
	* Try nanoGPT on a checkpoint.pt, see if the errors of \_dynamo are are persistent. -> Yup, no issue there! Have to debug more and see why.


* train more, see if you can hit the 1.4 mark? [YES]
	* when does the VL decompostion overfit?

* [URGENT] Code not working so far, try:  [DONE]
	* Check if backprop is working with torch.kron()
	* why is the backprop func only a view?

* VL decomp are zeros, investigate why [DONE], 
	* fixed.

* Write the initial code. [DONE]
	* Is it working? NO!

* Find (a not so random) way init. the weights  [DONE]
	* locate the checkpoints /  weights [DONE]
	* Decompose using the Van Loan Algorithm 

* Pre-training Investigation: [DONE]
	* You most likely need to pre-train the model, not just fine-tune
	* Is openWebText feasble? (It should be you got 4 A100 80BG)
	* Conclusion: Not really, no need.

### Ideas:	

* distill KronyBlocks. one by one, and store the checkpoint.

* prefer to drop in decompostion 1 by 1, i.e., allow the model to adapt to the new stucture.

* The mlp block takes considerable amount % of the total weights.:
	* decompose them first, make it work.
	* then do a style of attention. 

* Prune before decompositions (Why should this this should help, though)

* The money is in MLPs, attention blocks are perfect.
	* Leave them alone or at least change theem last.
	* Why: 
	*  1. they are pretty dense.
	*  2. most use flashattention which is implemented at the cuda level.

* Mixture of Experts as krone decomp? does it make sense?

 
### Details

* Repo structure now:
	* `train.py`
	* `model.py`
	* my ckeckpoint with kronecker decomposition, is saved in as ckpt2.pt
* I'm mostly using a single A100 80GB

### Check later:

* The .pynb for scaling/flops.
* MoE blog in Hugginface?


