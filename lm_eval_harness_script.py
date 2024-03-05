from transformers import GPT2LMHeadModel, GPT2Tokenizer
import lm_eval
import torch

device = "cuda:0"

tokenizer1  = GPT2Tokenizer.from_pretrained("gpt2")
model       = GPT2LMHeadModel.from_pretrained("./ww3")

model.to(device)

lm_eval.tasks.initialize_tasks() 
model_eval = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer = tokenizer1)

result  = lm_eval.evaluator.simple_evaluate(model_eval, tasks=["lambada"], batch_size=8, device = device)



print(result["results"])