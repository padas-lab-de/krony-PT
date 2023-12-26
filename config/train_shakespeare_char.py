# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'

eval_interval = 50 # keep frequent because we'll overfit
eval_iters = 20  # this one is used by estimate_loss()

max_iters = 1000 
lr_decay_iters = max_iters 



log_interval = 10 
# don't print too too often 
# I usually don't look at it, I'm setting the eval_iters low.. 
# so I freq log to wandb

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False
wandb_project = '10M-Early'
wandb_run_name = '1-Base-Model-smooth-lr'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
min_lr = 1e-4 # learning_rate / 10 usually


beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
warmup_iters = 50 # not super necessary potentially was 100




