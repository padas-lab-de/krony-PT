#$ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

eval_interval = 20   # sending to wandb. 
log_interval = 10    # mfu thingy
eval_iters = 20      # estimate_loss()
cut_the_run = 10000
block_size = 1024
gradient_accumulation_steps = 5*4

# lr stuff
learning_rate = 6e-4 # max learning rate
weight_decay = 1e-1
min_lr = 6e-5 
# min-lr, should be ~= learning_rate/10 per Chinchilla


# chinchilla recommendations.
max_iters = 600000 
warmup_iters = 1000 
lr_decay_iters = max_iters 


batch_size = 2

wandb_log = True 

init_from = "prune"
wandb_project = 'new_lr'
wandb_run_name= f"gpt2-{init_from}-lr-same-all-batch-1-warmup-1k"


