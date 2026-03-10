import torch
import torch.nn as nn
import numpy as np
import os

# ==========================================
# 1. KLE Core Functions (Math Components)
# ==========================================

def compute_eigenvalues(eta, variance, wn, device):
    """
    Compute eigenvalues (lambda) for the Karhunen-Loève Expansion.
    Formula: lambda_n = (2 * eta * variance) / (eta^2 * wn^2 + 1)
    """
    if not isinstance(eta, torch.Tensor):
        eta = torch.tensor(eta, device=device)
    if not isinstance(variance, torch.Tensor):
        variance = torch.tensor(variance, device=device)
    
    return (2 * eta * variance) / (eta**2 * wn**2 + 1)

def compute_eigenfunctions(eta, wn, z, L, device):
    """
    Compute eigenfunctions (phi) at spatial location z.
    Formula: phi_n(z) = factor * (eta * wn * cos(wn * z) + sin(wn * z))
    """
    if not isinstance(eta, torch.Tensor):
        eta = torch.tensor(eta, device=device)
    if not isinstance(L, torch.Tensor):
        L = torch.tensor(L, device=device)

    denominator = (eta**2 * wn**2 + 1) * L / 2.0 + eta
    factor = 1.0 / torch.sqrt(denominator)

    term1 = eta * wn * torch.cos(wn * z)
    term2 = torch.sin(wn * z)
    return factor * (term1 + term2)

# ==========================================
# 2. Neural Network Components
# ==========================================

def initialize_weights(model):
    """Xavier initialization (default gain=1)."""
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

def normalize_tz(t, z):
    """
    Normalize spatial-temporal inputs to the range [-1, 1].
    Assumes t in [0, 10] and z in [0, 99].
    """
    t_norm = 2 * (t / 10.0) - 1
    z_norm = 2 * (z / 99.0) - 1
    return t_norm, z_norm

class Encoder(nn.Module):
    """
    Auxiliary encoder layer to transform inputs before entering the main MLP.
    """
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.initialize_weights()

    def initialize_weights(self):
        initialize_weights(self)

    def forward(self, x):
        return self.activation(self.fc1(x))

class ModifiedMLP(nn.Module):
    """
    Modified MLP architecture with feature modulation using encoder outputs.
    Used for the state variable (psi) prediction.
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(ModifiedMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()
        self.initialize_weights()

    def initialize_weights(self):
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
    """
    Sub-network for residual correction (DNN part of the Hybrid model).
    """
    def __init__(self, input_dim, output_dim, hidden_dim=36):
        super(StandardMLP, self).__init__()
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
        self.initialize_weights_default()

    def initialize_weights_default(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. KLE Field Layer
# ==========================================

class KLEField(nn.Module):
    """
    A trainable layer representing the fluctuation of a stochastic field using KLE.
    Outputs only the fluctuation: sum_i sqrt(lambda_i) * phi_i(z) * xi_i
    """
    def __init__(self, mean, variance, eta, roots, L, device):
        super(KLEField, self).__init__()
        self.mean = mean
        self.variance = variance
        self.eta = eta
        self.L = L
        self.device = device

        self.register_buffer('wn', torch.tensor(roots, dtype=torch.float32, device=device).reshape(1, -1))
        self.nkl = self.wn.shape[1]

        # Xi (Random Coefficients) initialized to 0 (starts at mean field)
        self.xi = nn.Parameter(torch.zeros(1, self.nkl, device=device))

    def forward(self, z):
        eigenvalues = compute_eigenvalues(self.eta, self.variance, self.wn, self.device)
        sqrt_eigenvalues = torch.sqrt(eigenvalues)  # (1, nkl)

        phi_z = compute_eigenfunctions(self.eta, self.wn, z, self.L, self.device)  # (N, nkl)

        # Output fluctuation ONLY
        fluctuation = torch.sum(sqrt_eigenvalues * phi_z * self.xi, dim=1, keepdim=True)
        return fluctuation

# ==========================================
# 4. Main Hybrid PINN Model
# ==========================================

class PINN(nn.Module):
    def __init__(self, device):
        super(PINN, self).__init__()
        self.device = device
        self.L = 99.0

        # --- 1. State Network (Psi) ---
        self.encoder_1 = Encoder(2, 64)
        self.encoder_2 = Encoder(2, 64)
        self.state_net = ModifiedMLP(2, hidden_dim=64)

        # --- 2. Load Roots Internally ---
        roots_dir = 'C:/ORES_PINN/data/roots' # replace with the correct path
        
        def load_root_file(eta):
            path = os.path.join(roots_dir, f'eta={int(eta)}.npy')
            if not os.path.exists(path):
                raise FileNotFoundError(f"[Hybrid_PINN] Root file missing: {path}.")
            return np.load(path)

        roots_30 = load_root_file(30.0)
        roots_40 = load_root_file(40.0)
        roots_50 = load_root_file(50.0)

        # --- 3. Parameter Fields (Hybrid: KLE + DNN) ---
        nkl_Ks, nkl_alpha, nkl_n = 14, 11, 9

        self.kle_ln_Ks    = KLEField(3.2173, 0.5, 30.0, roots_30[:nkl_Ks],     self.L, device)
        self.kle_ln_alpha = KLEField(-3.3242, 0.3, 40.0, roots_40[:nkl_alpha],  self.L, device)
        self.kle_ln_n     = KLEField(0.4447, 0.1, 50.0, roots_50[:nkl_n],      self.L, device)

        self.net_ln_Ks    = StandardMLP(input_dim=1, output_dim=1)
        self.net_ln_alpha = StandardMLP(input_dim=1, output_dim=1)
        self.net_ln_n     = StandardMLP(input_dim=1, output_dim=1)

        # --- 4. Fixed VGM Constants ---
        self.theta_r = 0.078
        self.theta_s = 0.43

        self.register_buffer('mu_lnKs',    torch.tensor(3.2173))
        self.register_buffer('mu_lnalpha', torch.tensor(-3.3242))
        self.register_buffer('mu_lnn',     torch.tensor(0.4447))

        self.to(device)

    def forward(self, t, z):
        """
        Forward pass to compute states and reconstruct hybrid parameters.
        """
        # --- A. Predict State (Psi) ---
        t_norm, z_norm = normalize_tz(t, z)
        state_input = torch.cat([t_norm, z_norm], dim=-1)

        enc1_out = self.encoder_1(state_input)
        enc2_out = self.encoder_2(state_input)
        out_mlp = self.state_net(state_input, enc1_out, enc2_out)
        
        # Enforce negative pressure
        psi = -torch.exp(out_mlp)

        # --- B. Predict Parameters (Hybrid Reconstruction) ---
        # Note: Input to KLE is the physical z, input to DNN is normalized z
        
        # Ks: Mean + KLE Fluctuation + DNN Fluctuation
        fluct_kle_Ks = self.kle_ln_Ks(z)
        fluct_dnn_Ks = self.net_ln_Ks(z_norm)
        ln_Ks = self.mu_lnKs + fluct_kle_Ks + fluct_dnn_Ks

        # Alpha
        fluct_kle_alpha = self.kle_ln_alpha(z)
        fluct_dnn_alpha = self.net_ln_alpha(z_norm)
        ln_alpha = self.mu_lnalpha + fluct_kle_alpha + fluct_dnn_alpha

        # n
        fluct_kle_n = self.kle_ln_n(z)
        fluct_dnn_n = self.net_ln_n(z_norm)
        ln_n = self.mu_lnn + fluct_kle_n + fluct_dnn_n

        Ks = torch.exp(ln_Ks)
        alpha = torch.exp(ln_alpha)
        n = torch.exp(ln_n)

        # --- C. Compute Derived Variables (VGM) ---
        m = 1.0 - (1.0 / n)
        abs_psi = -psi
        
        se_term = 1.0 + (alpha * abs_psi).pow(n)
        Se = se_term.pow(-m)
        theta = self.theta_r + Se * (self.theta_s - self.theta_r)

        term1 = Se.pow(0.5)
        se_inv_m = 1.0 / se_term
        inner_term = torch.relu(1.0 - se_inv_m)
        term2 = (1.0 - inner_term.pow(m)).pow(2)

        K = Ks * term1 * term2

        return psi, theta, K, alpha, n, Ks