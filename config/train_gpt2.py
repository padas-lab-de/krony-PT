# Change the dims / path to the .pt file.
# torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

max_iters = 30000 
eval_interval = 100 # sending to wandb. 
log_interval = 100    # mfu thingy

eval_iters = 300      # estimate_loss()

block_size = 1024

#learning_rate = 6e-4 # max learning rate
learning_rate = 6e-5
weight_decay = 1e-1
min_lr = 6e-5
decay_lr = False   # learning rate fix

cut_the_run = max_iters 
gradient_accumulation_steps = 5
batch_size = 32

warmup_iters =  100
lr_decay_iters = max_iters 


wandb_log = False 
wandb_project = 'bias_included'

wandb_run_name= "1350_4_32"

dim1 = 768
dim2 = 768

init_from = "prune"
init_name = "imp-checks/gold_gold_4_32_iteration_1350.pt"
