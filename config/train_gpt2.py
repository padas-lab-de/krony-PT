# Change the dims / path to the .pt file.
# torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

max_iters = 200000 
eval_interval = 100   # sending to wandb. 
log_interval = 100    # mfu thingy
eval_iters = 50      # estimate_loss()

block_size = 1024

learning_rate = 6e-4 # max learning rate
weight_decay = 1e-1
min_lr = 6e-5 

cut_the_run = max_iters 
gradient_accumulation_steps = 2
batch_size = 32

warmup_iters =  500
lr_decay_iters = max_iters 


wandb_log = True 
wandb_project = 'bias_included'

# grad acc / batch size
wandb_run_name= "64_24_gpt2_bias_2_32"

dim1 = 64
dim2 = 24

init_from = "prune"
init_name = "out2/VL_64_24.pt"



