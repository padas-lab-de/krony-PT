#torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


eval_interval = 20   # sending to wandb. 
log_interval = 10    # mfu thingy
eval_iters = 20      # estimate_loss()
block_size = 1024

cut_the_run = 10000

# lr stuff
learning_rate = 6e-4 # max learning rate
weight_decay = 1e-1
min_lr = 6e-5 
# min-lr, should be ~= learning_rate/10 per Chinchilla


# 
gradient_accumulation_steps = 3*4
batch_size = 12

max_iters = 600000 
warmup_iters = 500 
lr_decay_iters = max_iters 


wandb_log = True 
wandb_project = 'reported'
wandb_run_name= "prune-small-batch-3-4-12-dist-0"

init_from = "prune"
init_name = "GPT2_prune_init_0_001.pt"


# poor man's logs
message = """
gradient_accumulation_steps = 1*4
batch_size = 20
"""

print(message)
