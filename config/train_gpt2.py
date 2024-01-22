#torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


eval_interval = 20   # sending to wandb. 
log_interval = 10    # mfu thingy
eval_iters = 20      # estimate_loss()
block_size = 1024
cut_the_run = 3000


# lr stuff
learning_rate = 1e-4 # max learning rate
weight_decay = 1e-1
min_lr = 6e-5 
# min-lr, should be ~= learning_rate/10 per Chinchilla


# 
gradient_accumulation_steps = 4*4
batch_size = 8

max_iters = 600000 
warmup_iters = 500 
lr_decay_iters = max_iters 


wandb_log = True 
wandb_project = 'reported'
wandb_run_name= f"first-run-VL"

init_from = "prune"
init_name = "GPT2_VL11.pt"


# poor man's logs
message = """
>>>> run log: \n
 init from VL.
 re-run of everything   \n
 3000 iterations is basically 10% of data  \n
 simple one, from pruning
"""

print(message)
