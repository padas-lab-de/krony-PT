# Change the dims / path to the .pt file.
# torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

max_iters = 5

eval_interval = 1    # sending to wandb. 
eval_iters = 400      # number of batches to consider in estimate_loss()

log_interval = 100    # mfu thingy
block_size = 1024

#learning_rate = 6e-4 # max learning rate
learning_rate = 6e-5
weight_decay = 1e-1
min_lr = 6e-4
decay_lr = False # learning rate fix

cut_the_run = max_iters 
gradient_accumulation_steps = 2
batch_size = 12

warmup_iters =  200
lr_decay_iters = max_iters 


wandb_log = False 
wandb_project = 'new_start'
wandb_run_name= "VL_256_1024_2"

#dim1 = 384
#dim2 = 3072
#factors = 1 

dim1 =  768
dim2 = 768
factors = 1

init_from = "prune"
init_name = "OG-checks/1350.pt"
