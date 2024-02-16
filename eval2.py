"""
---
title: Evaluation
summary: >
    Code to evaluate the model on NLP tasks through lm-evaluation-harness
---

# Evaluation

This is the code to test the model on
[EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

* [Evaluating half precision model on a single GPU](half_precision.html)
"""
from contextlib import redirect_stdout
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F

from lm_eval import tasks, evaluator, utils
from lm_eval.api.model import LM

#from lm_eval.base import BaseLM
from tokenizers import Tokenizer
from torch import nn
from tqdm import tqdm

#from labml import monit
#from labml_nn.neox.tokenizer import get_tokenizer


#class EvalHarnessAdapter(BaseLM):
class EvalHarnessAdapter(LM):
    """
    ## Evaluation Harness Adapter
    This is based on the [adapter from EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py)
    """

    def __init__(self, tokenizer: Tokenizer, vocab_size: int, batch_size: int):
        """
        :param tokenizer: is the [Huggingface Tokenizer](huggingface/tokenizers)
        :param vocab_size: is the size of the vocabulary
         (this differs from the tokenizer vocab size since neox adds some extra to make the embedding layer
         model parallel.)
        :param batch_size: is the batch size
        """
        super().__init__()
        self.tokenizer = tokenizer
        self._eot_token_id = self.tokenizer.eot_token
        self._vocab_size = vocab_size
        self._batch_size = batch_size

    @property
    def device(self):
        raise RuntimeError()

    @property
    def vocab_size(self):
        """Size of the vocabulary"""
        return self._vocab_size

    @property
    def eot_token_id(self):
        """End-of-text token"""
        return self._eot_token_id

    @property
    def max_length(self):
        """Maximum sequence length"""
        return 2048

    @property
    def max_gen_toks(self):
        """Maximum number of tokens to generate"""
        return 128

    @property
    def batch_size(self):
        """
        Batch size
        """
        return self._batch_size

    def tok_encode(self, string: str):
        """
        Encode a given text
        """
        return self.tokenizer.encode(string).ids

    #def tok_decode(self, tokens: List[int]):
    #    """
    #    Decode text from token ids
    #    """
    #    return self.tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor):
        raise NotImplementedError

    def _model_generate(self, context, max_length, eos_token_id):
        raise RuntimeError()

    def greedy_until(self, requests):
        raise RuntimeError()
    
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        return super().loglikelihood(requests)

    def generate_until(self, requests) -> List[str]:
        return super().generate_until(requests)
    
    def loglikelihood_rolling(self, requests) -> List[Tuple[float | bool]]:
        return super().loglikelihood_rolling(requests)
    
    @torch.no_grad()
    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        """
        ### Get log-likelihoods of the next tokens
        :param requests: List of requests containing the context and the expected continuation.
        :param disable_tqdm: If True, disable tqdm progress bar.
        """

        # For results
        res = []

        # Reorder the requests in the descending order of the lengths,
        # so that sequences with similar lengths are close
        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        reord = utils.Reorderer(requests, _collate)

        # Loop through requests with `batch_size` number of requests at a time
        for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
            
            inps = []             # To store the inputs for the batch
            continuations = []    # The continuations for the batch
            inplens = []          # Lengths of the input sequences
            padded_length = None  # Padded length for the batch
            
            # Loop through each request in the chunk and collect them into PyTorch tensors with paddings
            
            for _, context_enc, continuation_enc in chunk:
                # Concatenate
                inp = context_enc + continuation_enc
                # Truncate from left if the size exceeds the `max_length`
                inp = inp[-(self.max_length + 1):]
                # Remove final token
                inp = inp[:-1]
                # Create a tensor
                inp = torch.tensor(inp, dtype=torch.long)
                # Input length
                inplen = inp.shape[0]

                # Determine the padded length.
                # Shorter sequences will get padded.
                if padded_length is None:
                    padded_length = int(math.ceil(inplen / 32)) * 32
                # padded_length = padded_length if padded_length is not None else inplen

                # Padding
                padding = torch.zeros(padded_length - inplen, dtype=torch.long)

                # Add padding
                inp = torch.cat([inp, padding], dim=0)

                inps.append(inp)
                continuations.append(continuation_enc)
                inplens.append(inplen)

            # Get model logits
            logits = self._model_call(torch.stack(inps))

            # Get log softmaxes
            multi_logits = F.slog_softmax(logits, dim=-1)

        return res 
    
