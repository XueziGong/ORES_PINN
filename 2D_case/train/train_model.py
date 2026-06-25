import os
import sys
from pathlib import Path

import torch


BASE_DIR = Path(r"...\train") # replace with the correct path
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

from model_train import train_loop

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_MODULE_NAME = "KLE_PINN"
EXP_NAME = f"{MODEL_MODULE_NAME}_correct"
TARGET_SEEDS = [666, 667, 668, 669, 670]
EPOCHS = 100000
N_COLLOCATION = 10000
LR = 5e-4


if __name__ == "__main__":
    print("PyTorch:", torch.__version__)
    print("Device:", DEVICE)

    for seed in TARGET_SEEDS:
        train_loop(
            device=DEVICE,
            exp_name=EXP_NAME,
            train_seed=seed,
            model_module_name=MODEL_MODULE_NAME,
            epochs=EPOCHS,
            n_collocation=N_COLLOCATION,
            lr=LR,
        )
