import torch
import sys
import os

# Ensure current directory is in path
sys.path.append(os.getcwd())

# Import the training loop logic
from model_train import train_loop

# --- Configuration ---
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("PyTorch:", torch.__version__)
print("Device:", DEVICE)

exp_name = 'baseline'

# Define the specific list of seeds you want to iterate over
target_seeds = [666,667,668,669,670]

# Loop through the specific seeds list
for i, seed in enumerate(target_seeds):
    print(f"\n===========================================")
    print(f"   STARTING RUN {i+1}/{len(target_seeds)} (TRAIN SEED {seed})")
    print(f"===========================================")
    train_loop(
        DEVICE=DEVICE,
        exp_name=exp_name,
        train_seed=seed,       # Now uses the specific seed from the list
    )