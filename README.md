This is a detached fork of [NanoGPT @ Karpathy](https://github.com/karpathy/nanoGPT/) :goat:.

### ToC:
* [Goals](#goals)
* [Progress](#progress)
* [TODO](#todo)
* [DONE](#done)
* [Misc](#misc)

---
### Goals: <a name="goals">

Main goal: Getting under 3.00 loss on owt with Kronecker Products and under 300 steps, approx. takes 30 min. (starting from the hf checkpoint)

1. Factorize single weight matrices into a product of Kroneckers.
2. Test scaling of distillation / training. With different strategies.
3. Test impact of adding multiple Kroneckers factors.
4. Test if weight freezing has any significance to post-training or distillation.

---
### **Progress**  <a name="progress">

* Now, I exclusively work on GPT2 now, and I'm testing different setups.
* 50% deterministic (either pick even or odd rows) prunning works way better than any other initialization.
* Main thing to be done next: Update the optimizer, and have different learning rates for new params and already trained parameters.

* [Link to wandb logs](https://wandb.ai/benayad7/freezing-test?workspace=user-benayad7)
* [New experiments](https://wandb.ai/benayad7/new_lr?workspace=user-benayad7)

<!---
* [Link to pdf (soon)](https://wandb.ai/benayad/shakespeare-char?workspace=user-sunnyayoub17)
--->


* Why move to GPT2: I feel like the small model (10M param  with characters) is very unreliable. I can literally get the model to do anything I want with more training. I'll just switch all my focus on GPT2 124M model. And only play with the other one for prototyping.

**Some remarks:** 

* GPT2 takes approx 20GB of memory. 
* Initial loss is 10.9886 10.9898. with random init.
* HF checkpoint is 3.11 approx (on open web text)

---
### **TODO:** <a name="todo">

* Setup -- GPT2 eval with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

* Prototyping on Shakespeare -- evening kinda fun. 
	* Small thing increments.
	* Load everything from config script [DONE]
	* Save some checkpoints, save them locally. [DONE]
	* See how everything is changing overtime.

* Things to try:
	* bring all weights with same lr.. don't differentiate btw old and new. [DONE]
	* different learning rates [DONE / In progress]
	* different batch size, maybe it's taking too long for an update to happen.. [DONE]
	* in prune init, remove 0, put 0.0001 instead. >> I think this should take priority. there must be smth better than 0.
	* try a schedule of 600000k for the KP weights. >>>>> This ASAAAAAP [DONE]
	* Try the random baseline, random inits as 
	* mask the `w_1` in forward pass to 1.
	* 

* Need to start retraining for more time. At least 1 epoch. [ so far, I train for 0.5%, in the paper they train for 10%]

* 2 factors with prunning:
	* decompose and test correctness.
	* reproduce loss?

* Write results on freezing on .tex file. (this could wait a bit) >> this is not very interesting.

* Write code for 1 by 1 training:  [Code is done, testing not fully]
	* form the highest level to the lowest and vice versa.
	* Log results.
	* Add **ddp** to code for 1 by 1 training (`train_distillation.py`). (this is done for train.py, so it should be am easy copy paste, approx 2 hours of work)

* Get a 3 decompositions, target point is **85M** (same or under DistilGPT)
	* Decomposition 1 > 95M from $(3072, 768)$ to $(3072, 384)$  [DONE]

* In the distillation, have a quick/efficient way of:
	* Parameters loading.
	* Right now, I'm setting params manually
	* this should be done auto, be a professional! it's better for long term dev.

* Optimize compute [No use for it now, gpu are doing their job as of now.] 
	* Still haven't used any KP props. But I doubt that it would help [compute is not the bottleneck here].

* More Experiments.
	* different n / m for KroneckerDecom
	* Try the Kronecker decomposition using the lib.

* Add a quick script on how to generate a new checkpoint from scratch. 

---
*  Questions:
	* Freezing the weights apparently helps, is there a way to quantify the impact?
	* How the KP factors are changing with and without freezing of other weights?
	* Some metric applied on the grads? 

* **Experiments that needs to be done:** >> Please check the latex document. Sec X.X (I'll make this public soon)

* How can you monitor what is your network learning?
	* We need some logging of the gradient / activation maps (I think they're called attention maps, not to be confused with attention blocks of the transformer).
	* Start with this [blog by Karpathy](http://karpathy.github.io/2019/04/25/recipe/)

* Add this to your tex file.  Van Loan Kronecker Decomposition is actually not the closest to W:
	* i.e., when I distill the student/teacher. the mse btw the original weights and KP weights actually gets bigger. this is kinda surprising, but it could make sense somehow.
	* next: add grad accumulation (I doubt that It would help) [DONE], true, didn't help.
	* also add grad-clipping [DONE]
	* Investigate this more. [DONE]
	* Van Loan is a shitty init (make this friendly). 

### **DONE**    <a name="done">

* Investigate batch_size and nproc_per_node impact on n_tokens and %epoch [DONE]
	* write a section on it.
	* tell me exactly, Epoch = f( function batch x gradient_acc x n_nodes )
	* with the current setup, 4 A100s. Each step is 250k tokens. and train data is 9B tokens. 
	* To see 1% of the data: 

* Work on **Optimizer/lr**: [DONE]
	* I need to investigate the learning rates. Cause apparently, results are not very stable.
	* Make it easy way to set different  lr for diff group of variables.
	* Groups as: pre-trained weights and newly introduced ones. So we can log differences...

* 2 factors decomposition: [DONE]
	* Decompose / Test if the loss values are the same.
	* Test if every approx is correct!

* Fix distributed testing. **ddp** (you shouldn't have deleted dumb ass) [DONE] (but not working, most likely a mem. issue)
	* Need to able to run on two nodes, today, better, yesterday!
	* Fixed the issue: add `CUDA_VISIBLE_DEVICES` accordingly, and `nproc_per_node`

* Test Freezing, for Rand, VL and Prune initialization. [DONE]
	* Test mixed strategies: (Freeze / Release) x Repeat.

* Automate Kronecker decomposition, one single file that generates the factors and checkpoint and stores it as *ckpt_n_n_fac* **[DONE]**
	* one script should run from terminal to generate the ckpt
	* please fix the splitting asap, c_fc and c_proj should have opposite terms.

* Write code for multiple Kron Products factors:  [DONE]
	* you have to make it easier to **load** the state weights from outside.
	* (Soon, I want to make it end2end decompose-training-evaluate.)

* Get the training curve for the new 3 inits, for a few iterations: [DONE]
	1. random
	2. VL
	3. Simple prunning trick. 
	--> prune first. only (keep one of the two)

* GPT 124M -- Initial setup:  [DONE]
	* Loading / Testing with multiple nodes. [DONE]
	* What is webtext? download / play with. [DONE]
	* Reproducing the loss [DONE]
	* Compare with original [DONE]

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
	* Conclusion: remove `torch.compile()` // I also have not noticed any difference in training speed.

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

### Code:

* Do not use torch compile.



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
