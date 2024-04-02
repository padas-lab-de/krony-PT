# Change the dims / path to the .pt file.
# torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

max_iters = 10000

# wandb 
wandb_log = False #wandb_log = True 
eval_interval = 5
wandb_project = 'new_start'
wandb_run_name= "256_64_10_batch_6_12"
# wandb 

eval_iters = 20      # number of batches to consider in estimate_loss()

log_interval = 100    # mfu thingy
block_size = 1024

# lr

learning_rate = 6e-5
weight_decay = 1e-1
#min_lr = 6e-4
min_lr = 6e-5
decay_lr = False 
# lr 

cut_the_run = max_iters 
gradient_accumulation_steps = 10
batch_size = 32

warmup_iters =  100
lr_decay_iters = max_iters 

init_from = "else"
init_name = "./check2/95M_prune_init_iteration_90600.pt"

