# Change the dims / path to the .pt file.
# torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

max_iters = 50000 
eval_interval = 10   # sending to wandb. 
log_interval = 100    # mfu thingy
eval_iters = 50      # estimate_loss()

block_size = 1024

#learning_rate = 6e-5 # max learning rate
#weight_decay = 1e-1
#min_lr = 6e-5 

learning_rate = 6e-5 # max learning rate
weight_decay = 1e-1
min_lr = 6e-5 

cut_the_run = max_iters 
gradient_accumulation_steps = 10
batch_size = 16

warmup_iters =  500
lr_decay_iters = max_iters 


wandb_log = False
wandb_project = 'bias_included'

# grad acc / batch size
wandb_run_name= "gold_gold_10_16"

dim1 = 768
dim2 = 768

init_from = "prune"
init_name = "checkpt2/gold_gold_4_32_iteration_1350.pt"


### try with even smaller back size but higher grad acc
###  



