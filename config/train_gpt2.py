# Change the dims / path to the .pt file.
# torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

max_iters = 30000 
eval_interval = 50   # sending to wandb. 
log_interval = 100    # mfu thingy

eval_iters = 50      # estimate_loss()

block_size = 1024

learning_rate = 6e-4 # max learning rate
weight_decay = 1e-1
min_lr = 6e-5

cut_the_run = max_iters 
gradient_accumulation_steps = 6
batch_size = 28

warmup_iters =  100
lr_decay_iters = max_iters 


wandb_log = True 
wandb_project = 'bias_included'

wandb_run_name= "1350_6_28"

dim1 = 768
dim2 = 768

init_from = "prune"
init_name = "imp-checks/gold_gold_4_32_iteration_1350.pt"
