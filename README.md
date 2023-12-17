This is a a detached fork of [NanoGPT @ Karpathy](https://github.com/karpathy/nanoGPT/) :goat:.

The goal is to factorize single weight matrices into a product of Kronecker Matrices. 

---
### Progress

Status: Initial code (seems like It) is working for MLP decomposition.

**Some reporting:**

* [Link to wandb logs](https://wandb.ai/benayad/shakespeare-char?workspace=user-sunnyayoub17)

* Original nanoModel is 10.7M, Facftorized model in 4.3M.
* KD is slow? (not really), but  robust to overfitting.
	* Q: Is it very robust or do the gradients get saturated? hence, no updates? 

---
### **TODO:**

* Write code for distilled initialization of KronyMLP [IN-PROGRESS]
	* Initial code not working, **TRY**:
	* 1.  add batch norm, see if it helps.
	* 2. instead of random in/out, try actual x,y from data.
	* Test/Compare improved KP computations.

* Investigate the `torch.\_dynamo` error:
	* run the 5M mini-mini gpt
	* why does it happen only a few times.
	* run the experiments without compiling the model, and see if there are any change to the outcome

* I'm very sus of gradients saturation, please see how you can monitor that asap.
	* would torch profilers help?

* Optimize compute 
	* Still haven't used any KP props. But I doubt that it would help.

* More Experiment.
	* different n / m for KroneckerDecom
	* Try the Kronecker decompostion using the lib.

* Clean repo, 
	* remove unnecessary if/else for readablity.

* MoE? what is it? can we decompose it?

### **DONE**

* Report using wandb, log everything. [DONE]

* Fix the resume "training"
	* fix the iter_num in .pt files. [DONE]
	* I only want to pick the weights (state_dict()) everything else should be same as Training from scratchi [DONE]
	* Try nanoGPT on a checkpoint.pt, see if the errors of \_dynamo are are persistent. -> Yup, no issue there! Have to debug more and see why.

* train more, see if you can hit the 1.4 mark? [YES]
	* when does the VL decompostion overfit?

* [URGENT] Code not working so far, try:  [DONE]
	* Check if backprop is working with torch.kron() [looks fine]
	* why is the backprop func only a view?

* VL decomp are zeros, investigate why [DONE], 
	* fixed.

* Write the initial code. [DONE]
	* Is it working? NO!

* Find (a not so random) way to init. the weights  [DONE]
	* [Van Loan Algorithm](https://link.springer.com/chapter/10.1007/978-94-015-8196-7_17)
	* [See section 2.2 of this paper](https://zarmesh.com/wp-content/uploads/2019/05/An-efficient-method-to-solve-large-linearizable-inverse-pr_2019_Computers-.pdf)

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
	* Why avoid touching attention blocks: 
	*  1. they are pretty dense.
	*  2. most use flashattention which is implemented at the cuda level.

* Prune before decompositions (Why should this this should help, though)

* Mixture of Experts as krone decomp? does it make sense?
 
### Details

* Repo structure now:
	* `train.py`
	* `model.py`
	* my ckeckpoint with kronecker decomposition, is saved in as ckpt2.pt
* I'm mostly using a single A100 80GB


