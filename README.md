## ToC:
* [Description.](#Abstract)
* [Results of 81M and 95M.](#Results)
* [Train and Evaluate new models.](#play)
    * [Get the data.](#data)
    * [Train new models in 3 steps.](#Train)
    * [Evaluation.](#Eval)


---
## Description:<a name="Abstract"> 

We reduce the size of GPT2 by compressing the MLPs weights (practically two thirds? of the model) of each attention layer with Kronecker Products. Effectively down-sizing GPT2 from 124M parameters to two variants, the 81M and 95M parameters models. Compared to other papers:

* We don't use distillation (empirical evidence shows no benefits compared to only Vanilla supervised learning).
* We have a systematic compression scheme, i.e., compress all layers the same way. We don't see any reasons to why we would only compress odd layers (besides to "force" the number of parameters to be under certain threshold).
* We use weight tying, i.e., the embedding and softmax  matrices are **identical**. This makes our model, the only "effective" 81M model.
* We try different compression schemes (67M, 81M, and 95M)
    * We propose a new simple initialization for the 95M model. 
    * We use multiple factors for the VL based inits.

---
## Results:<a name="results"> 

We train 3 classes of models, 67M (being the smallest we can get), 81M (mid size), and 95M (highest model we can get). Below are some "numbers".

### 81M model:

Krony-PT (81M) outperforms DistilGPT (82M) on all benchmarks, and especilally on the Lambada dataset.

| # Params  | Model            | wikitext-103 | wikitext-2 | Lambada |
| --- | --- | --- | --- | --- |
| 124M      | GPT2              | 29.16        | 24.67      | 45.28      |
| 82M       | DistilGPT2        | 44.53        | 36.48      | 76.00      |
| 81M       | **KronyPT-81M-1350**  | **41.98**        | **34.99**      | -          |
| 81M       | **KronyPT-81M-3950**  | -            | -          | **64.92**      |

Our 81M model performs on par with other Kronecker based models (x,y,z papers), while having **39M parameters** less. Even outperforming KnGPT on Lambada.


| # Params  | Model            | wikitext-103 | wikitext-2 | Lambada |
| --- | --- | --- | --- | --- |
| 81M       | **KronyPT-81M-1350**  | 41.98        | 34.99      | -          |
| 81M       | **KronyPT-81M-3950**  | -            | -          | 64.92      |
| 119M(*)   | TQCompressedGPT2  | 40.28        | 32.25      | 64.72      |
| 119M(*)   | KnGPT-2 (Huawei)  | 40.97        | 32.81      | 67.62      |


### 95M model:

Here we compare different initialization strategies: Van Loan (VL) and a (new) prunning based init. (add the results for the prune based method).

| Model       | wikitext-103 | wikitext-2 | Lambada |
|----------   |----------|----------|----------|
| 95M - VL    | 41.80    | 35.50   |  61.34    | 
| 95M - prune | _  | _  | _ |


---
## How to play: <a name="play"> 
### Get the data: <a name="data"> 
1. Clone the repository.
2. Create the data: check `./data/owt/prepare.py` 
> We solely use Open Web Text (owt) for training. 

### Train new models: 3 steps.<a name="Train"> 

1. Generate a valid Kronecker decomposition: use script `kron_decompose.py`, specify the dimensions, and number of factors.

```
$ python kron_decompose.py dim_1 dim_2 n_factors
```

2. Update your training configuration at `./config/train_gpt2`
	* Change training specifications to your need.


3. Train the model:

```
$ python train.py config/train_gpt2.py
```

4. After training, you should have a checkpoint (say `./checks/my_checkpoint.pt`) ready for evaluation.
### Evaluation: <a name="eval"> 

Assuming you have your Kronecker checkpoint stored at `./checks/my_checkpoint.pt`

1. Convert the `KronyPT`  to a GPT-like format.
```
$ python krony_to_gpt.py  ./path/to/check.pt  output_dir 
```

This would convert your `KronyPT` model to a suitable GPt-like format stored at `out_dir`. The dimensions and number of factors are inferred directly from the checkpoint, hence no need to be provided.

2. Test the perplexity for `wikitext` and `lambada`:

```
$ python perplexity_eval.py output_dir wiki103
$ python perplexity_eval.py output_dir 			# to evaluate on all 3 datasets
```

You have 3 options: `wiki103`, `wiki1`, `lambada`. Not specifying which dataset would return the perplexity for all 3.

### Progress is moved to progress.md

* Check the file: `script/progress.md` 
* Add link to report. 
