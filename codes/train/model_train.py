import torch
import torch.nn as nn
import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 0. Global Configuration & Path Setup
# ==========================================
# Use relative paths for the open-source project structure
MODEL_DIR = 'C:/ORES_PINN/codes/model/' 
BASE_DATA_PATH = 'C:/ORES_PINN/data/obs'
COORDS_PATH = 'C:/ORES_PINN/data'
RESULTS_DIR = 'C:/ORES_PINN/results'

# Import the PINN model
sys.path.append(MODEL_DIR)
from ORES_PINN import PINN

# ==========================================
# 1. Post-Training Prediction & Save
# ==========================================
def save_predictions(pinn, DEVICE, save_path):
    """
    Generate and save the final spatial-temporal field predictions 
    and the 1D parameter profiles (VGM parameters).
    No evaluation against ground truth is performed here.
    """
    print("\n--- Starting Post-Training Prediction ---")
    pinn.eval()
    
    # Define Physical Grid for Prediction
    z_phys_eval = np.linspace(99, 0, 100) 
    t_phys_eval = np.linspace(0, 10, 101)
    
    T_mesh_phys, Z_mesh_phys = np.meshgrid(t_phys_eval, z_phys_eval)
    
    # Flatten grids for model input
    t_flat = torch.tensor(T_mesh_phys.flatten(), dtype=torch.float32).reshape(-1, 1).to(DEVICE)
    z_flat = torch.tensor(Z_mesh_phys.flatten(), dtype=torch.float32).reshape(-1, 1).to(DEVICE)
    
    # --- Predict Spatial-Temporal Fields ---
    with torch.no_grad():
        psi_pred, theta_pred, K_pred, _, _, _ = pinn(t_flat, z_flat)
        
    shape_grid = T_mesh_phys.shape
    pred_psi_grid   = psi_pred.cpu().numpy().reshape(shape_grid)
    pred_theta_grid = theta_pred.cpu().numpy().reshape(shape_grid)
    pred_K_grid     = K_pred.cpu().numpy().reshape(shape_grid)
    
    # --- Predict 1D Parameter Profiles (VGM Parameters) ---
    z_profile_phys = np.linspace(99, 0, 100)
    z_profile = torch.tensor(z_profile_phys, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
    t_dummy = torch.zeros_like(z_profile)  # Parameters are time-independent
    
    with torch.no_grad():
        _, _, _, alpha_prof, n_prof, Ks_prof = pinn(t_dummy, z_profile)
    
    # Log-transform parameters for standardized saving
    pred_ln_alpha = np.log(alpha_prof.cpu().numpy().flatten())
    pred_ln_n     = np.log(n_prof.cpu().numpy().flatten())
    pred_ln_Ks    = np.log(Ks_prof.cpu().numpy().flatten())
    
    # --- Save Predictions to Disk ---
    print(f"Saving final predictions to {save_path}...")
    np.save(os.path.join(save_path, 'pred_theta.npy'), pred_theta_grid)
    np.save(os.path.join(save_path, 'pred_psi.npy'), pred_psi_grid)
    np.save(os.path.join(save_path, 'pred_K.npy'), pred_K_grid)
    
    np.save(os.path.join(save_path, 'pred_lnKs.npy'), pred_ln_Ks)
    np.save(os.path.join(save_path, 'pred_lnalpha.npy'), pred_ln_alpha)
    np.save(os.path.join(save_path, 'pred_lnn.npy'), pred_ln_n)
    print("Predictions saved successfully.")

# ==========================================
# 2. Main Training Loop
# ==========================================
def train_loop(DEVICE, exp_name, train_seed, epochs=100000):
    # Physical domain bounds
    T_MAX_PHYS = 10.0  
    Z_MAX_PHYS = 99.0  
    
    save_path = os.path.join(RESULTS_DIR, exp_name, f'train_seed_{train_seed}')
    os.makedirs(save_path, exist_ok=True)
        
    print(f"--- Experiment: {exp_name} (Train Seed {train_seed}) ---")
    
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

    # --- 1. Load Observation Data (Noisy) ---
    print("Loading observation data...")
    try:
        obs_theta_raw = np.load(os.path.join(BASE_DATA_PATH, 'obs_theta.npy')) 
        obs_psi_raw = np.load(os.path.join(BASE_DATA_PATH, 'obs_psi.npy'))       
        time_coords = np.load(os.path.join(COORDS_PATH, 'time_coords.npy'))
        
        # Define sensor locations
        idx_theta = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95] 
        idx_psi   = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        
        z_theta_phys = Z_MAX_PHYS - np.array(idx_theta, dtype=float)
        z_psi_phys   = Z_MAX_PHYS - np.array(idx_psi, dtype=float)
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # --- 2. Prepare Training Tensors ---
    def prepare_training_tensors(obs_data, z_locs_phys, t_vals_phys):
        t_list, z_list, val_list = [], [], []
        n_sensors = obs_data.shape[0]
        n_times = min(obs_data.shape[1], len(t_vals_phys))

        for i in range(n_sensors):
            z_val_phys = z_locs_phys[i]
            for j in range(n_times):
                t_list.append(t_vals_phys[j])
                z_list.append(z_val_phys)
                val_list.append(obs_data[i, j])
        
        return (torch.tensor(t_list, dtype=torch.float32).reshape(-1, 1).to(DEVICE),
                torch.tensor(z_list, dtype=torch.float32).reshape(-1, 1).to(DEVICE),
                torch.tensor(val_list, dtype=torch.float32).reshape(-1, 1).to(DEVICE))

    t_obs_theta, z_obs_theta, v_obs_theta = prepare_training_tensors(obs_theta_raw, z_theta_phys, time_coords)
    t_obs_psi, z_obs_psi, v_obs_psi = prepare_training_tensors(obs_psi_raw, z_psi_phys, time_coords)
    
    # --- 3. Initialize Model ---
    pinn = PINN(device=DEVICE).to(DEVICE)
    
    optimizer = torch.optim.Adam(pinn.parameters(), lr=5e-4)
    mse_loss = nn.MSELoss()
    
    n_collocation = 4000
    
    # Loss Weights
    w_theta = 1.0
    w_psi = 0.01
    w_pde = 1.0
    
    loss_history = []
    loss_theta_history = []
    loss_psi_history = []
    loss_pde_history = []
    
    def grad(outputs, inputs):
        """Helper function to compute gradients for the PDE loss."""
        return torch.autograd.grad(outputs, inputs,
                                   grad_outputs=torch.ones_like(outputs),
                                   create_graph=True)[0]

    print("Starting main training loop...")
    pbar = tqdm(range(epochs))
    
    for epoch in pbar:
        pinn.train()
        optimizer.zero_grad()
        
        # --- Data Loss ---
        _, pred_theta_obs, _, _, _, _ = pinn(t_obs_theta, z_obs_theta)
        loss_theta = mse_loss(pred_theta_obs, v_obs_theta)
        
        pred_psi_obs, _, _, _, _, _ = pinn(t_obs_psi, z_obs_psi)
        loss_psi = mse_loss(pred_psi_obs, v_obs_psi)
        
        # --- Physics (PDE) Loss ---
        t_col = torch.rand(n_collocation, 1, device=DEVICE) * T_MAX_PHYS
        z_col = torch.rand(n_collocation, 1, device=DEVICE) * Z_MAX_PHYS
        
        t_col.requires_grad = True
        z_col.requires_grad = True
        
        psi_col, theta_col, K_col, _, _, _ = pinn(t_col, z_col)
        
        dtheta_dt = grad(theta_col, t_col) 
        dpsi_dz   = grad(psi_col, z_col)    
        dK_dz     = grad(K_col, z_col)      
        d2psi_dz2 = grad(dpsi_dz, z_col)    
        
        lhs = dtheta_dt
        rhs = dK_dz * (dpsi_dz + 1.0) + K_col * d2psi_dz2
        
        loss_pde = torch.mean(torch.square(lhs - rhs))
        
        # --- Optimization ---
        loss = w_theta * loss_theta + w_psi * loss_psi + w_pde * loss_pde
        
        loss.backward()
        optimizer.step()
        
        # --- Logging ---
        loss_history.append(loss.item())
        loss_theta_history.append(loss_theta.item())
        loss_psi_history.append(loss_psi.item())
        loss_pde_history.append(loss_pde.item())
        
        pbar.set_description(f"T:{loss.item():.2e}|Th:{loss_theta.item():.2e}|Psi:{loss_psi.item():.2e}|PDE:{loss_pde.item():.2e}")

    # --- 4. Save Final Results ---
    print("\nTraining completed. Saving model and evaluating...")
    torch.save(pinn.state_dict(), os.path.join(save_path, 'model_final.pth'))
    
    # Save loss histories
    np.save(os.path.join(save_path, 'loss_history.npy'), loss_history)
    np.save(os.path.join(save_path, 'loss_theta_history.npy'), loss_theta_history)
    np.save(os.path.join(save_path, 'loss_psi_history.npy'), loss_psi_history)
    np.save(os.path.join(save_path, 'loss_pde_history.npy'), loss_pde_history)
    
    # Plot and save loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Total Loss', alpha=0.7)
    plt.plot(loss_theta_history, label='Theta Loss', alpha=0.7)
    plt.plot(loss_psi_history, label='Psi Loss', alpha=0.7)
    plt.plot(loss_pde_history, label='PDE Loss', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(save_path, 'loss_curve.png'), dpi=300)
    plt.close()
    
    # Generate and save final parameter and state predictions
    save_predictions(pinn, DEVICE, save_path)
    
    print(f"\nExperiment {exp_name} (Train Seed {train_seed}) Completed successfully.")

# ==========================================
# Script Entry Point
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")