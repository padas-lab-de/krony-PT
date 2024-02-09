import torch
import torch.nn as nn

from torch.nn import functional as F
from model_origin import GPT

class CustomGeneration():
	def __init__(self, model, tokenizer, conf):
		self.model = model
		self.tokenizer = tokenizer
		self.device = torch.device("cuda")
		self.config = conf
		self.model.eval()

	def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
		"idx is the prompt encoded, returns the ids, have to be decoded back"
		for _ in range(max_new_tokens):
			idx_cond = idx if idx.size(1) <= self.config["block_size"] else idx[:, -self.config["block_size"]:]
			logits, _ = self.model(idx_cond)
			logits = logits[:, -1, :] / temperature
			if top_k is not None:
				v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
				logits[logits < v[:, [-1]]] = -float('Inf')
			probs = F.softmax(logits, dim=-1)
			idx_next = torch.multinomial(probs, num_samples=1)
			idx = torch.cat((idx, idx_next), dim=1)
		return idx

	def generate_until(self, requests):
		outputs = []
		for request in requests:
			input_str, generation_params = request.args[0], request.args[1]
			input_ids = torch.tensor([self.tokenizer(input_str, add_special_tokens=True)]).to(self.device)
			max_new_tokens = generation_params.get("max_gen_toks", 128)
			temperature = generation_params.get("temperature", 1.0)
			top_k = generation_params.get("top_k", None)
			outputs.append(self._generate(input_ids, max_new_tokens, temperature, top_k))
		return outputs

	def loglikelihood(self, requests):
		log_probs = []
		is_greedy = []
		for request in requests:
			input_str, target_str = request.args[0], request.args[1]
			input_ids = torch.tensor([self.tokenizer(input_str, add_special_tokens=True)]).to(self.device)
			target_ids = torch.tensor([self.tokenizer(target_str, add_special_tokens=False)]).to(self.device)
			logits, _ = self.model(input_ids, labels=target_ids[:, :-1])
			log_probs_t = torch.gather(logits, 2, target_ids[:, 1:].unsqueeze(2)).squeeze()
			log_probs.append(-log_probs_t.sum().item())
			is_greedy.append(torch.argmax(logits[:, -1, :], dim=-1).eq(target_ids[:, -1]).item())
		return [(lp, ig) for lp, ig in zip(log_probs, is_greedy)]

	def loglikelihood_rolling(self, requests):
		log_probs = []
		for request in requests:
			input_str = request.args[0]
			input_ids = torch.tensor([self.tokenizer(input_str, add_special_tokens=True)]).to(self.device)
			eos_index = self.tokenizer.eos_token_id
			logits, _ = self.model(input_ids)
			log_probs_t = torch.logsumexp(logits[:, -1, :] - torch.log(torch.tensor([self.model.config.vocab_size], device=self.device)), dim=-1)
			log_probs.append(-log_probs_t.item())
		return [(lp,) for lp in log_probs]


"""
Here, we define a MyCustomLM class that takes your GPT2 model as input 
and initializes it in evaluation mode. 

We also define a _generate method that takes care of generating
the text using the same logic as in your original generate method. 

Finally, we define the generate_until method that takes a list of Instance objects, extracts the input 
string and generation parameters from each Instance, and generates the text using the _generate method.

Note that in the _generate method, I assumed that your model expects input in the form of a batch of 
token IDs, so I wrapped the input IDs in a batch with size 1 using unsqueeze(0). 

Also, I assumed that your tokenizer is a PreTrainedTokenizer instance from the Hugging Face transformers library.
If these assumptions are not correct, you'll need to modify the code accordingly.



class MyCustomLM(GPT2LMHeadModel):
    def __init__(self, model_path, max_new_tokens=100, temperature=1.0, top_k=None, eot_token="\n\n", max_gen_toks=128):
        super().__init__(model_path)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.config.model_name_or_path)
        self.config = self.tokenizer.model_init().config
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.eot_token = eot_token
        self.max_gen_toks = max_gen_toks

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        # Calculate loglikelihood for each request
        results = []
        for request in requests:
            input_ids = request.input_ids
            target = request.target
            with torch.no_grad():
                output = self(input_ids, labels=target)
                loss = output.loss
                ll = output.logits.softmax(-1).log()
                is_greedy = torch.argmax(output.logits, dim=-1).item() == target.argmax()
                results.append((ll.item(), is_greedy))
        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        # Calculate loglikelihood for each request
        results = []
        for request in requests:
            input_ids = request.input_ids
            with torch.no_grad():
                output = self(input_ids)
                ll = output.logits.softmax(-1).log()
                is_greedy = torch.argmax(output.logits, dim=-1).item() == request.target.argmax()
                results.append((ll.item(), is_greedy))
        return results

# Example usage
model_path = "path/to/your/model"
instance = Instance(input="Hello", request_type="generate_until", args={"num_beams": 1, "max_length": 50})
my_custom_lm = MyCustomLM(model_path, max_new_tokens=100, temperature=1.0, top_k=None, eot_token="\n\n", max_gen_toks=128)

requests = [instance]
generated_texts = my_custom_lm.generate_until(requests)
loglikelihoods = my_custom_lm.loglikelihood(requests)
loglikelihoods = [ll[0] for ll


Ok, so:

* requests -->  List[Instance]  // basic : requests = [one_req], now, what is one_req
* one_req, is one request, gotcha haha. 
* 

"""
