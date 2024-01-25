#torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

max_iters = 600000 

eval_interval = 20   # sending to wandb. 
log_interval = 10    # mfu thingy
eval_iters = 20      # estimate_loss()
block_size = 1024


learning_rate = 6e-4 # max learning rate
weight_decay = 1e-1
min_lr = 6e-5 

cut_the_run = 10000
gradient_accumulation_steps = 3*3
batch_size = 12

warmup_iters = 10000 
lr_decay_iters = max_iters 


wandb_log = True 
wandb_project = 'distil'
wandb_run_name= "prune-init-3-3-12-cosine-lr"

init_from = "prune"
init_name = "out/GPT2_prune_init_0_001.pt"


#init_name  = "checkpoints/prune-small-batch-5-4-12_iteration_8100.pt"
