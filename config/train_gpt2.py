#$ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

eval_interval = 20   # sending to wandb. 
log_interval = 10    # mfu thingy
eval_iters = 20      # estimate_loss()

block_size = 1024
gradient_accumulation_steps = 5*4*4

# lr stuff
learning_rate = 1e-4 # max learning rate
weight_decay = 1e-1
min_lr = 6e-5 
# min-lr, should be ~= learning_rate/10 per Chinchilla


# chinchilla recommendations.
max_iters = 600000 
warmup_iters = 2000 
lr_decay_iters = max_iters 

batch_size = 2
cut_the_run = 30000

wandb_log = True 

wandb_project = 'new_lr'
wandb_run_name= f"gpt2-prune-new_init_money"


init_from = "prune"
init_name = "GPT2_prune_init_0_001.pt"

# here i cha
# small batch size, small warm up and smaller learnig rate

# >> I removed weight_decay for all other weights, and brought w_decay to 0.1 for KP weights
# >>> long warm up 3k, weight decay pushed to 1e-2, 
# >>> this run is regarding pruning -- with 1 and -0.001
