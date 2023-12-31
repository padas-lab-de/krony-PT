This is a detached fork of [NanoGPT @ Karpathy](https://github.com/karpathy/nanoGPT/) :goat:.

### ToC:
* [Goals](#goals)
* [Progress](#progress)
* [TODO](#todo)
* [DONE](#done)
* [Misc](#misc)

---
### Goals: <a name="goals">

1. Factorize single weight matrices into a product of Kroneckers.
2. Test scaling of distillation / training. With different strategies.
3. Test impact of adding multiple Kroneckers factors.
4. Test if weight freezing has any significance to post-training or distillation.

---
### **Progress**  <a name="progress">

**Status:** I'm moving to GPT2 124M. 

* [Link to pdf (soon)](https://wandb.ai/benayad/shakespeare-char?workspace=user-sunnyayoub17)
* [Link to wandb logs](https://wandb.ai/benayad/shakespeare-char?workspace=user-sunnyayoub17)

**Update:** 

I feel like the small model (10M param  with characters) is very unreliable. I can literally get the model to do anything I want with more training. I can't trust. I'll just switch all my focus on GPT2 124M model. And only play with the other one for prototyping.

---
### **TODO:** <a name="todo">

* Write an end 2 end training / eval framework: (embrace the HF power.)
	* what benchmarks are used in the paper? 
	* reproduce the paper's results.

* Optimizer: I need to be re-write it, because I intend to have different lr for diff group of variables.
	* Need a way to split between, pre-trained weights and newly introduced ones. So we can log differences...

* Write code for multiple Kron Products factors:  [In Progress]
	* you have to make it easier to **load** the state weights from outside.
	* (Soon, I want to make it end2end decompose-traning-evaluate.)
	* (the less interventions the better)

* Automate Kronecker decomposition, one single file that generates the factors and checkpoint and stores it as *ckpt_n_n_fac* **[DONE]**
	* one script should run from terminal to generate the ckpt
	* please fix the splitting asap, cfc and cproj should have opposite terms.

*  Questions:
	* Freezing the weights apparently helps, is there a way to quantify the impact?
	* How the KP factos are changing with and without freezing of other wieghts?
	* Some metric applied on the grads? 


* change the behavior of optimizers, mainly the lr:	
	* currently even the pre-trained params are set to the same lr as the other decomposed matrices. 
	* doesn't seem right.

* **Experiments that needs to be done:** >> Please check the latex document. Sec X.X

* How can you monitor what is your network learning?
	* we need some serious logging of the gradients. 
	* start with this [blog by Karpathy](http://karpathy.github.io/2019/04/25/recipe/)

* Kronecker Decompostion is actually not the closest to W:
	* i.e., when I distill the student/teacher. the mse btw the original weights and KP weights actually gets bigger. this is kinda surprising, but it could make sense somehow.
	* I should be able to  the val loss, if that one decreases while the mse increases, then that's fine.
	* next: add grad accumulation (I doubt that It would help) 
	* also add grad-clipping
	* this is  (potentially) a good point.. 
	* Investigate this more.

* In the distillation, have a quick/efficient way of:
	* parameters loading.
	* right now, I'm setting params manually
	* this should be done auto, be a professional! it's better for long term dev.

* Optimize compute 
	* Still haven't used any KP props. But I doubt that it would help.

* More Experiment.
	* different n / m for KroneckerDecom
	* Try the Kronecker decompostion using the lib.

* Clean this repo
	* remove unnecessary if/else for readability
	* bring back ddp, dumb ass.
	* clean the TODO/DONE. only keep necessary stuff.
	* e.g., add toc
	* add a quick script on how to generate a new checkpoint from scratch.

### **DONE**    <a name="done">


* GPT 124M -- Initial setup:  [DONE]
	* Loading / Testing with multiple nodes. [DONE]
	* What is webtext? download / play with. [DONE]
	* Reproduing the loss [DONE]
	* compare with original [DONE]

* (As I suspected) when I freeze all the weights. And only train the new plugged in weights. 
	* The network doesn't learn anything. [Wrong]
	* Plot twist: Not really, optimized had a bug, now fixed. Freezing helps activating more.
	* But it's an interesting direction that I will tackle in the following iterations.

* Write code for distilled initialization of KronyMLP [DONE]
	* Initial code (random i\o) not working, **TRY**:
	* 1.  add batch norm, see if it helps.
	* 2. instead of random in/out, try actual x,y from data. 
	* Test/Compare improved KP computations.

* MoE? what is it? can we decompose it? [DONE]
	* Conclusion: Good stuff.
	* Next: loading Mixtral 47B params. playing with it.

* Investigate the `torch._dynamo` error: [DONE]
	* run the 5M mini-mini gpt [DONE]
	* run the experiments without compiling the model, and see if there are any change to the outcome [DONE: works fine, I'll drop the /compile for now]
	* Conclusion: remove `torch.compile()` // I also have not notived any difference in training speed.

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

<a name="misc">
### Ideas:	

* Prefer to drop in decomposition 1 by 1, i.e., allow the model to adapt to the new structure, both during training and distillation.
	* This thing for distillation is done.
	* Implementing this for training is not easy (at least from a first try).
	* Try: 
	1. load two GPT krony and GPT, 
	2. decompose them into nn.Module, similar to what you have done in distillation
	3. each %k of iteration, make training flow in one layer


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
	* my ckeckpoint with kronecker decomposition, is saved in as ckpt2.pt
* I'm mostly using a single A100 80GB

* **How to decompose and generate a new checkpoint**
	* Completely handled by `kronDecomp.py` (change the name of  this file)
