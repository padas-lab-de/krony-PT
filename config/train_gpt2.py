#$ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

eval_interval = 20   # sending to wandb. 
log_interval = 10    # mfu thingy
eval_iters = 20      # estimate_loss()

block_size = 1024
gradient_accumulation_steps = 5*4

# lr stuff
learning_rate = 1e-3 # max learning rate
weight_decay = 1e-2
min_lr = 6e-5 
# min-lr, should be ~= learning_rate/10 per Chinchilla


# chinchilla recommendations.
max_iters = 600000 
warmup_iters = 3000 
lr_decay_iters = max_iters 

batch_size = 16
cut_the_run = 30000

wandb_log = True 

wandb_project = 'new_lr'
wandb_run_name= f"gpt2-prune-new_init_1"


init_from = "prune"
init_name = "out/GPT2_prune_init_1.pt"

# >>> long warm up 3k, weight decay pushed to 1e-2, 
# >>> this run is regarding pruning -- with 1 and -1
