## How to play: (not completed)
1. Clone the directory.
2. Create the data: check `./data/owt/prepare.py` 


### Train new models: 3 steps.

1. Generate the Kronecker decomposition: use script `kron_decompose.py`, specify the dimensions, and number of factors.
```
$ python kron_decompose dim_1 dim_2 n_factors
```
2. Fill the configuration file (`config/train_gpt2`), with the appropriate details for training and for the models specs. 

3. Train.
```
$ python train.py config/train_gpt2.py
```


2. Update your training configuration at `config/train_gpt2`
	* Add the adequate dimensions and factors / wandb link / where you have stored the model

3. Run: `python train.py config/train_gpt2`

### Evaluation: 

Assuming you have your Kronecker checkpoint stored at `./check/my_checkpoint.pt`

1. Convert the `KronyPT`  to a GPT-like format.
```
$ python krony_to_gpt.py  ./path/to/check.pt  output_dir dim1 dim2 factors
```
This would convert you `KronyPT` model to a suitable GPt-like format stored at `out_dir`. Make sure you provide the exact dimensions (dim1, dim2) and factors.

2. Test the perplexity for `wikitext` and `lambada`:

```
$ python perplexity.py output_dir wiki103
$ python perplexity.py output_dir 			# to evaluate on all 3 datasets

```
You have 4 options: `wiki103`, `wiki1`, `lambada`. The option `all` would return the perplexity for all datasets.






## Progress is moved to progress.md

* Old README.md got bloated >> Link, check `progress.md`
* Initial report >> Link.
