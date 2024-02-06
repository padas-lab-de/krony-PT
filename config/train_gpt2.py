# Change the dims / path to the .pt file.
# torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

max_iters = 600000 
eval_interval = 20   # sending to wandb. 
log_interval = 10    # mfu thingy
eval_iters = 20      # estimate_loss()
block_size = 1024
learning_rate = 6e-4 # max learning rate
weight_decay = 1e-1
min_lr = 6e-5 

cut_the_run = 30000
gradient_accumulation_steps = 8*3
batch_size = 8

warmup_iters =  500
lr_decay_iters = max_iters 


wandb_log = True 
wandb_project = 'new_decompostions'

wandb_run_name= "starting_from_128_32_iteration_29700"

dim1 = 128
dim2 = 32 

init_from = "prune"
init_name = "checkpoints/128_32_iteration_29700.pt"



