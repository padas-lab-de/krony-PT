# Change the dims / path to the .pt file.
# torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

max_iters = 10000 
eval_interval = 50   # sending to wandb. 
log_interval = 100    # mfu thingy
eval_iters = 50      # estimate_loss()

block_size = 1024

learning_rate = 6e-5 # max learning rate
weight_decay = 1e-1
min_lr = 6e-5

cut_the_run = max_iters 
gradient_accumulation_steps = 4
batch_size = 32

warmup_iters =  100
lr_decay_iters = max_iters 


wandb_log = True 
wandb_project = 'bias_included'

wandb_run_name= "highest_run_4_32"

dim1 = 768
dim2 = 768

init_from = "prune"
init_name = "check2/768_768_emb_plug_iteration_29997.pt"
