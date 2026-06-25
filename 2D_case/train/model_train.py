import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# replace with the correct path
BASE_DIR = Path(r"...\train")
MODEL_DIR = BASE_DIR
OBS_DATA_DIR = BASE_DIR / "obs"
GRID_DATA_DIR = Path(r"...\data_raw")
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

sys.path.insert(0, str(MODEL_DIR))


def import_pinn_class(model_module_name):
    module = importlib.import_module(model_module_name)
    return module.PINN


def load_grid_or_default():
    times_path = GRID_DATA_DIR / "times.npy"
    x_grid_path = GRID_DATA_DIR / "x_grid.npy"
    z_grid_path = GRID_DATA_DIR / "z_grid.npy"

    times = np.load(times_path) if times_path.exists() else np.linspace(0.0, 10.0, 101)
    x_grid = np.load(x_grid_path) if x_grid_path.exists() else np.linspace(0.0, 10.0, 11)
    z_grid = np.load(z_grid_path) if z_grid_path.exists() else np.linspace(0.0, 99.0, 100)

    return times.astype(float), x_grid.astype(float), z_grid.astype(float)


def load_or_build_sensor_coords(kind, z_max_phys):
    x_file = BASE_DIR / f"{kind}_free_x_coords.npy"
    z_file = BASE_DIR / f"{kind}_free_z_coords.npy"

    if x_file.exists() and z_file.exists():
        return np.load(x_file).astype(float), np.load(z_file).astype(float)

    x_obs = np.array([3.0, 5.0, 7.0], dtype=float)
    if kind == "theta":
        depths = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95], dtype=float)
    elif kind == "psi":
        depths = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
    else:
        raise ValueError(f"Unknown observation type: {kind}")

    z_obs = z_max_phys - depths
    x_coords = np.repeat(x_obs, len(z_obs))
    z_coords = np.tile(z_obs, len(x_obs))
    return x_coords, z_coords


def prepare_training_tensors(obs_data, x_locs, z_locs, t_vals, device):
    obs_data = np.asarray(obs_data, dtype=float)
    n_sensors = obs_data.shape[0]
    n_times = min(obs_data.shape[1], len(t_vals))

    if len(x_locs) != n_sensors or len(z_locs) != n_sensors:
        raise ValueError(
            "Observation data and coordinate files are inconsistent. "
            f"obs sensors={n_sensors}, x coords={len(x_locs)}, z coords={len(z_locs)}."
        )

    t_arr = np.tile(t_vals[:n_times], n_sensors)
    x_arr = np.repeat(x_locs, n_times)
    z_arr = np.repeat(z_locs, n_times)
    v_arr = obs_data[:, :n_times].reshape(-1)

    return (
        torch.tensor(t_arr, dtype=torch.float32, device=device).reshape(-1, 1),
        torch.tensor(x_arr, dtype=torch.float32, device=device).reshape(-1, 1),
        torch.tensor(z_arr, dtype=torch.float32, device=device).reshape(-1, 1),
        torch.tensor(v_arr, dtype=torch.float32, device=device).reshape(-1, 1),
    )


def grad(outputs, inputs):
    value = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return torch.zeros_like(inputs) if value is None else value


def train_loop(
    device,
    exp_name,
    train_seed,
    model_module_name="OR_PINN",
    epochs=100000,
    n_collocation=10000,
    lr=5e-4,
):
    save_path = CHECKPOINT_DIR / exp_name / f"train_seed_{train_seed}"
    save_path.mkdir(parents=True, exist_ok=True)

    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

    times, x_grid, z_grid = load_grid_or_default()
    t_max = float(np.max(times))
    x_max = float(np.max(x_grid))
    z_max = float(np.max(z_grid))

    obs_theta_path = OBS_DATA_DIR / "obs_theta.npy"
    obs_psi_path = OBS_DATA_DIR / "obs_psi.npy"
    if not obs_theta_path.exists():
        raise FileNotFoundError(f"Missing observation file: {obs_theta_path}")
    if not obs_psi_path.exists():
        raise FileNotFoundError(f"Missing observation file: {obs_psi_path}")

    obs_theta = np.load(obs_theta_path)
    obs_psi = np.load(obs_psi_path)

    theta_x, theta_z = load_or_build_sensor_coords("theta", z_max)
    psi_x, psi_z = load_or_build_sensor_coords("psi", z_max)

    t_obs_theta, x_obs_theta, z_obs_theta, v_obs_theta = prepare_training_tensors(
        obs_theta, theta_x, theta_z, times, device
    )
    t_obs_psi, x_obs_psi, z_obs_psi, v_obs_psi = prepare_training_tensors(
        obs_psi, psi_x, psi_z, times, device
    )

    PINN = import_pinn_class(model_module_name)
    pinn = PINN(device=device).to(device)

    optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    w_theta = 1.0
    w_psi = 0.01
    w_pde = 1.0

    print(f"Experiment: {exp_name}")
    print(f"Model: {model_module_name}")
    print(f"Seed: {train_seed}")
    print(f"Checkpoint path: {save_path}")
    print(f"obs_theta: {obs_theta.shape}, obs_psi: {obs_psi.shape}")

    for epoch in tqdm(range(epochs), desc="Training"):
        pinn.train()
        optimizer.zero_grad()

        _, theta_pred, _, _, _, _ = pinn(t_obs_theta, x_obs_theta, z_obs_theta)
        psi_pred, _, _, _, _, _ = pinn(t_obs_psi, x_obs_psi, z_obs_psi)

        loss_theta = mse_loss(theta_pred, v_obs_theta)
        loss_psi = mse_loss(psi_pred, v_obs_psi)

        t_col = torch.rand(n_collocation, 1, device=device) * t_max
        x_col = torch.rand(n_collocation, 1, device=device) * x_max
        z_col = torch.rand(n_collocation, 1, device=device) * z_max
        t_col.requires_grad_(True)
        x_col.requires_grad_(True)
        z_col.requires_grad_(True)

        psi_col, theta_col, K_col, _, _, _ = pinn(t_col, x_col, z_col)

        dtheta_dt = grad(theta_col, t_col)
        dpsi_dx = grad(psi_col, x_col)
        dpsi_dz = grad(psi_col, z_col)
        dK_dx = grad(K_col, x_col)
        dK_dz = grad(K_col, z_col)
        d2psi_dx2 = grad(dpsi_dx, x_col)
        d2psi_dz2 = grad(dpsi_dz, z_col)

        rhs_x = dK_dx * dpsi_dx + K_col * d2psi_dx2
        rhs_z = dK_dz * (dpsi_dz + 1.0) + K_col * d2psi_dz2
        loss_pde = torch.mean((dtheta_dt - rhs_x - rhs_z) ** 2)

        loss = w_theta * loss_theta + w_psi * loss_psi + w_pde * loss_pde
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            tqdm.write(
                f"Epoch {epoch + 1}: "
                f"loss={loss.item():.4e}, "
                f"theta={loss_theta.item():.4e}, "
                f"psi={loss_psi.item():.4e}, "
                f"pde={loss_pde.item():.4e}"
            )

    torch.save(pinn.state_dict(), save_path / "model_final.pth")
    print(f"Saved model parameters to: {save_path / 'model_final.pth'}")
