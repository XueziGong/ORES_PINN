import os

import numpy as np
import torch
import torch.nn as nn


def compute_eigenvalues(eta, variance, wn, device):
    if not isinstance(eta, torch.Tensor):
        eta = torch.tensor(eta, dtype=torch.float32, device=device)
    if not isinstance(variance, torch.Tensor):
        variance = torch.tensor(variance, dtype=torch.float32, device=device)
    return (2.0 * eta * variance) / (eta**2 * wn**2 + 1.0)


def compute_eigenfunctions(eta, wn, z, L, device):
    if not isinstance(eta, torch.Tensor):
        eta = torch.tensor(eta, dtype=torch.float32, device=device)
    if not isinstance(L, torch.Tensor):
        L = torch.tensor(L, dtype=torch.float32, device=device)

    denominator = (eta**2 * wn**2 + 1.0) * L / 2.0 + eta
    factor = 1.0 / torch.sqrt(denominator)
    return factor * (eta * wn * torch.cos(wn * z) + torch.sin(wn * z))


def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def normalize_txz(t, x, z):
    t_norm = 2.0 * (t / 10.0) - 1.0
    x_norm = 2.0 * (x / 10.0) - 1.0
    z_norm = 2.0 * (z / 99.0) - 1.0
    return t_norm, x_norm, z_norm


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()
        initialize_weights(self)

    def forward(self, x):
        return self.activation(self.fc1(x))


class ModifiedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()
        initialize_weights(self)

    def forward(self, x, enc1_out, enc2_out):
        h = self.activation(self.fc1(x))
        h = h * enc1_out + (1.0 - h) * enc2_out
        h = self.activation(self.fc2(h))
        h = h * enc1_out + (1.0 - h) * enc2_out
        h = self.activation(self.fc3(h))
        h = h * enc1_out + (1.0 - h) * enc2_out
        h = self.activation(self.fc4(h))
        h = h * enc1_out + (1.0 - h) * enc2_out
        return self.fc5(h)


class StandardMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=36):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        initialize_weights(self)

    def forward(self, x):
        return self.net(x)


def get_roots_dir():
    local_roots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roots")
    if os.path.isdir(local_roots_dir):
        return local_roots_dir
    return r"...\data\roots" # replace with the correct path


class KLEField(nn.Module):
    def __init__(self, mean, variance, eta, roots, L, device):
        super().__init__()
        self.mean = mean
        self.variance = variance
        self.eta = eta
        self.L = L
        self.device = device
        self.register_buffer(
            "wn",
            torch.tensor(roots, dtype=torch.float32, device=device).reshape(1, -1),
        )
        self.xi = nn.Parameter(torch.zeros(1, self.wn.shape[1], device=device))

    def forward(self, z):
        eigenvalues = compute_eigenvalues(self.eta, self.variance, self.wn, self.device)
        phi_z = compute_eigenfunctions(self.eta, self.wn, z, self.L, self.device)
        fluctuation = torch.sum(torch.sqrt(eigenvalues) * phi_z * self.xi, dim=1, keepdim=True)
        return fluctuation, phi_z


class PINN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.L = 99.0

        self.encoder_1 = Encoder(3, 64)
        self.encoder_2 = Encoder(3, 64)
        self.state_net = ModifiedMLP(3, hidden_dim=64)

        roots_dir = get_roots_dir()

        def load_roots(eta):
            path = os.path.join(roots_dir, f"eta={int(eta)}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Root file not found: {path}")
            return np.load(path)

        roots_30 = load_roots(30.0)

        self.kle_ln_Ks = KLEField(3.2173, 0.5, 30.0, roots_30[:14], self.L, device)
        self.kle_ln_alpha = KLEField(-3.3242, 0.3, 30.0, roots_30[:14], self.L, device)
        self.kle_ln_n = KLEField(0.4447, 0.1, 30.0, roots_30[:14], self.L, device)

        self.net_ln_Ks = StandardMLP(1, 1)
        self.net_ln_alpha = StandardMLP(1, 1)
        self.net_ln_n = StandardMLP(1, 1)

        self.theta_r = 0.078
        self.theta_s = 0.43

        self.register_buffer("mu_lnKs", torch.tensor(3.2173, dtype=torch.float32))
        self.register_buffer("mu_lnalpha", torch.tensor(-3.3242, dtype=torch.float32))
        self.register_buffer("mu_lnn", torch.tensor(0.4447, dtype=torch.float32))

        self.gamma_logits_Ks = nn.Parameter(torch.tensor(-2.1972, dtype=torch.float32, device=device))
        self.gamma_logits_alpha = nn.Parameter(torch.tensor(-2.1972, dtype=torch.float32, device=device))
        self.gamma_logits_n = nn.Parameter(torch.tensor(-2.1972, dtype=torch.float32, device=device))

        self.to(device)

    @torch.no_grad()
    def get_trainable_weights(self):
        return {
            "gamma_Ks": torch.sigmoid(self.gamma_logits_Ks).item(),
            "gamma_alpha": torch.sigmoid(self.gamma_logits_alpha).item(),
            "gamma_n": torch.sigmoid(self.gamma_logits_n).item(),
        }

    def _orthogonalize(self, dnn_raw, phi_z, gamma_logits):
        dot_d_phi = torch.sum(dnn_raw * phi_z, dim=0, keepdim=True)
        dot_phi_phi = torch.sum(phi_z * phi_z, dim=0, keepdim=True)
        projection = torch.sum((dot_d_phi / (dot_phi_phi + 1e-8)) * phi_z, dim=1, keepdim=True)
        gamma = torch.sigmoid(gamma_logits)
        return dnn_raw - (1.0 - gamma) * projection

    def forward(self, t, x, z):
        t_norm, x_norm, z_norm = normalize_txz(t, x, z)
        state_input = torch.cat([t_norm, x_norm, z_norm], dim=-1)

        enc1_out = self.encoder_1(state_input)
        enc2_out = self.encoder_2(state_input)
        psi = -torch.exp(self.state_net(state_input, enc1_out, enc2_out))

        fluct_kle_Ks, phi_z_Ks = self.kle_ln_Ks(z)
        fluct_dnn_Ks = self._orthogonalize(self.net_ln_Ks(z_norm), phi_z_Ks, self.gamma_logits_Ks)
        ln_Ks = self.mu_lnKs + fluct_kle_Ks + fluct_dnn_Ks

        fluct_kle_alpha, phi_z_alpha = self.kle_ln_alpha(z)
        fluct_dnn_alpha = self._orthogonalize(self.net_ln_alpha(z_norm), phi_z_alpha, self.gamma_logits_alpha)
        ln_alpha = self.mu_lnalpha + fluct_kle_alpha + fluct_dnn_alpha

        fluct_kle_n, phi_z_n = self.kle_ln_n(z)
        fluct_dnn_n = self._orthogonalize(self.net_ln_n(z_norm), phi_z_n, self.gamma_logits_n)
        ln_n = self.mu_lnn + fluct_kle_n + fluct_dnn_n

        Ks = torch.exp(ln_Ks)
        alpha = torch.exp(ln_alpha)
        n = torch.exp(ln_n)

        m = 1.0 - 1.0 / n
        se_term = 1.0 + (alpha * (-psi)).pow(n)
        Se = se_term.pow(-m)

        theta = self.theta_r + Se * (self.theta_s - self.theta_r)
        inner_term = torch.relu(1.0 - 1.0 / se_term)
        K = Ks * Se.pow(0.5) * (1.0 - inner_term.pow(m)).pow(2)

        return psi, theta, K, alpha, n, Ks
