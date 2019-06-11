import os

root_dir = os.path.expanduser("~")
root_dir = os.path.join(root_dir, "Desktop")

train_data_path = os.path.join(root_dir, "CFSM/quora/chunked/train_*")
eval_data_path = os.path.join(root_dir, "CFSM/quora/chunked/val_*")
decode_data_path = os.path.join(root_dir, "CFSM/quora/hunked/test_*")
vocab_path = os.path.join(root_dir, "CFSM/quora/vocab")
cluster_path = os.path.join(root_dir, "CFSM/quora/paths")
log_root = os.path.join(root_dir, "CFSM/log")

save_model_iter = 1000
max_iterations = 1000000
use_gpu=True

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 8
sample_size= 4
max_enc_steps= 20
max_dec_steps= 20
beam_size= 8
min_dec_steps= 5
vocab_size= 25000
optimizer='adagrad'
lr=1e-5
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0
eps = 1e-12