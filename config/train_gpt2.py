# Change the dims / path to the .pt file.
# torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

max_iters = 100000

eval_interval = 100    # sending to wandb. 
eval_iters = 50      # number of batches to consider in estimate_loss()

log_interval = 100    # mfu thingy
block_size = 1024

#learning_rate = 6e-4 # max learning rate
learning_rate = 6e-5
weight_decay = 1e-1
#min_lr = 6e-4
min_lr = 6e-5
decay_lr = False # learning rate fix

cut_the_run = max_iters 
gradient_accumulation_steps = 6
batch_size = 24

warmup_iters =  200
lr_decay_iters = max_iters 


wandb_log = True 
wandb_project = 'new_start'
wandb_run_name= "256_64_10_batch_6_12"

dim1 = 256
dim2 = 64
factors = 10

#dim1 = 96
#dim2 = 24


init_from = "prune"
init_name = "./VLs/VL_256_64_10.pt"
#init_name = "./check2/ff/VL_384_3072_1_iteration_99990.pt"


