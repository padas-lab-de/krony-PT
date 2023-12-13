This is a a detached fork from [NanoGPT@Karpathy](https://github.com/karpathy/nanoGPT/)

The goal is to factorize single weight matrices into a product of Kronecker Matrices. 

Detailed Tasks:

1. Find a not so dumb init of the weights  [DONE]
	*. locate the checkpoints /  weights [DONE]
	*. Decompose using the Van Loan Algorithm

2. Pre-training Investigation: [DONE]
	* You most likely need to pre-train the model, not just fine-tune
	* Is openWebText feasble? (It should be you got 4 A100 80BG)
	* Conclusion: Not really, no need.

3. Loss 101:
	1. locate the loss
	2. make one simple change to one matrix
	3. see how everything behaves.
	4. a way to log loss training.
	5. a working pipeline, so you save time later on (Karpathy probb has this set-up already, he's that kind of guy) 

* Drop in the kronecker decompositions
	* drop one by one, let the model "get the vibe" of the new weights/architecture.

* Is it working:
	1. Yes? Good, optimize compute now. and utilize the KP rules (A x B)x
	2. No? you're f'ed, think of a better idea.


Ideas:	
* The mlp take considerable amount.:
	* the money is in the mlp weights.
	* decompose them first, make it work.
	* then do a style of attention. 

 
Scary questions:
* what if to compute kron., libraries usually extend the memory... hence, you're using the same weights/even more.

Important points from original readme:

* The file `train.py` reproduces GPT-2 (124M) on **OpenWebText**, 
* Running on a single 8XA100 40GB node in about 4 days of training. 

* The code itself is plain and readable: 
	* `train.py` is a ~300-line boilerplate training loop
	* `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

> Ideas:

* Prune before docompositions.

* The money is in MLPs, attention blocks are perfect, 
	* Leave them alone or at least change em last.
	* Why: 1. they are pretty dense. 2. most use flashattention which is implemented at the cuda level.

* Mixture of Experts as krone decomp >> that's the money dawg.

* For details on the original implementation, please refer to the original README by Karpathy :goat:.
