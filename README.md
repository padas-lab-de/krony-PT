This is a a detached fork of :goat: [NanoGPT @ Karpathy](https://github.com/karpathy/nanoGPT/)

The goal is to factorize single weight matrices into a product of Kronecker Matrices. 

### Detailed Tasks:

1. Find (a not so random) way init. the weights  [DONE]
	* locate the checkpoints /  weights [DONE]
	* Decompose using the Van Loan Algorithm 

2. Pre-training Investigation: [DONE]
	* You most likely need to pre-train the model, not just fine-tune
	* Is openWebText feasble? (It should be you got 4 A100 80BG)
	* Conclusion: Not really, no need.

3. Write the initial code. [DONE]


TODO:

4. Write code for distilled init
---
* Is it working:
	1. Yes? Good, optimize compute now. and utilize the KP rules (A x B)x
	2. No? you're f'ed, think of a better idea.


### Ideas:	

* distill KronyBlocks. one by one, and store the checkpoint.

* I prefer to drop in decompostion 1 by 1, and each time allow the model to adapt to the new stucture.

* The mlp take considerable amount % of the total weights.:
	* decompose them first, make it work.
	* then do a style of attention. 

* Prune before decompositions (Why should this this should help, though)

* The money is in MLPs, attention blocks are perfect.
	* Leave them alone or at least change theem last.
	* Why: 
	*  1. they are pretty dense.
	*  2. most use flashattention which is implemented at the cuda level.

* Mixture of Experts as krone decomp.

 
Some details regading this repo:

* The file `train.py` reproduces GPT-2 (124M) on **OpenWebText**, 
* Running on a single 8XA100 40GB node in about 4 days of training. 

* Repo structure now:
	* `train.py`
	* `model.py`
	* my ckeckpoint with kronecker decomposition, is saved in as ckpt2.ptA

> Check later:

* The .pynb for scaling/flops.
* MoE blog in Hugginface?


* Check the scaling reporting (I deleted) that was done with the original repo. 
