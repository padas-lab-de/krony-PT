# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB

# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True 
wandb_project = 'new_lr'
wandb_run_name='gpt2-VL-init-speed-lr'


# 12 batch size * 1024 block size * 5 gradaccum * 4 GPUs = 250k tokens per iter
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5*4

# lr stuff
learning_rate = 6e-2 # max learning rate
weight_decay = 1e-1
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla


# n_iterations
max_iters = 10000 
warmup_iters = 10 #how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla

# eval stuff
eval_interval = 20   
#was 1000 this one is for traning logging and logging to wandb

log_interval = 10   
# used for logging in the mfu part of the loop



eval_iters = 20     #was 200 this one is for inside estimate_loss