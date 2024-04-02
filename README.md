## ToC:

* Project (#Project):
    * Abstract (#Abstract):
    * Results of 85M and 95M (#Results):
* How to play (#play):
    * Get the data (#data):
    * Train new models: 3 steps (#Train):
    * Evaluation (#Evaluation):
    * Progress is moved to progress.md (#Progress)


---
## Project: 
### Abstract:
### Some early results:
### 85M model:
### 95M model:

| Model       | Column 2 | Column 3 | Column 4 |
|----------   |----------|----------|----------|
| 95M - VL    | Row 1.2  | Row 1.3  | Row 1.4  |
| 95M - prune | Row 2.2  | Row 2.3  | Row 2.4  |


Detailed report can be found here >>

---
## How to play: 
### Get the data: 
1. Clone the repository.
2. Create the data: check `./data/owt/prepare.py` 
> We solely use Open Web Text (owt) for training. 


### Train new models: 3 steps.

1. Generate a valid Kronecker decomposition: use script `kron_decompose.py`, specify the dimensions, and number of factors.
```
$ python kron_decompose dim_1 dim_2 n_factors
# dim_1 dim_2 ...
```

2. Update your training configuration at `./config/train_gpt2`
	* Add the adequate dimensions and factors, and training specifications.


3. Train the model:
```
$ python train.py config/train_gpt2.py
```

4. After training, you should have a checkpoint (say `./checks/my_checkpoint.pt`) ready for evaluation.
### Evaluation: 

Assuming you have your Kronecker checkpoint stored at `./checks/my_checkpoint.pt`

1. Convert the `KronyPT`  to a GPT-like format.
```
$ python krony_to_gpt.py  ./path/to/check.pt  output_dir 
```

This would convert your `KronyPT` model to a suitable GPt-like format stored at `out_dir`. The dimensions and number of factors are inferred directly from the checkpoint, hence no need to be provided.

2. Test the perplexity for `wikitext` and `lambada`:

```
$ python perplexity.py output_dir wiki103
$ python perplexity.py output_dir 			# to evaluate on all 3 datasets

```
You have 4 options: `wiki103`, `wiki1`, `lambada`. The option `all` would return the perplexity for all datasets.

### Progress is moved to progress.md

* link progress page here
* link report above and delete this line
