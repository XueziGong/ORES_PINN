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
    
    # Normalization factor
    denominator = (eta**2 * wn**2 + 1) * L / 2.0 + eta
    factor = 1.0 / torch.sqrt(denominator)
    
    # z is physical coordinate here. Shape: z(N, 1), wn(1, N_KL) -> Broadcasts to (N, N_KL)
    term1 = eta * wn * torch.cos(wn * z)
    term2 = torch.sin(wn * z)
    
    return factor * (term1 + term2)

# ==========================================
# 2. Neural Network Components
# ==========================================

def initialize_weights(model):
    """Xavier initialization for network weights."""
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

# ==========================================
# 3. KLE Field Layer
# ==========================================

class KLEField(nn.Module):
    """
    A trainable layer representing a stochastic field using KLE.
    Learns the coefficients 'xi' instead of direct field values.
    """
    def __init__(self, mean, variance, eta, roots, L, device):
        super(KLEField, self).__init__()
        self.mean = mean
        self.variance = variance
        self.eta = eta
        self.L = L
        self.device = device
        
        # Roots (omega_n) are fixed constants
        self.register_buffer('wn', torch.tensor(roots, dtype=torch.float32, device=device).reshape(1, -1))
        
        # Number of KL terms
        self.nkl = self.wn.shape[1]
        
        # Xi (Random Coefficients) are the TRAINABLE parameters
        # Initialized to zeros to start from the Mean Field
        self.xi = nn.Parameter(torch.zeros(1, self.nkl, device=device))

    def forward(self, z):
        """
        Reconstruct the field at spatial location z.
        Formula: Field(z) = Mean + sum( sqrt(lambda) * phi(z) * xi )
        """
        # 1. Compute Eigenvalues (Lambda)
        eigenvalues = compute_eigenvalues(self.eta, self.variance, self.wn, self.device)
        sqrt_eigenvalues = torch.sqrt(eigenvalues)
        
        # 2. Compute Eigenfunctions (Phi) using physical z
        phi_z = compute_eigenfunctions(self.eta, self.wn, z, self.L, self.device)
        
        # 3. KLE Summation over KL terms
        fluctuation = torch.sum(sqrt_eigenvalues * phi_z * self.xi, dim=1, keepdim=True)
        
        # 4. Add Mean
        return self.mean + fluctuation

# ==========================================
# 4. Main KLE-PINN Model
# ==========================================

class PINN(nn.Module):
    def __init__(self, device):
        """
        Args:
            device (torch.device): Compute device (CPU or CUDA).
            
        Note:
            This model loads required roots (.npy files) internally.
            Ensure the directory 'roots_dir' contains the necessary files.
        """
        super(PINN, self).__init__()
        self.device = device
        self.L = 99.0  # Domain Length
        
        # --- 1. State Network (Psi) ---
        self.encoder_1 = Encoder(2, 64)
        self.encoder_2 = Encoder(2, 64)
        self.state_net = ModifiedMLP(2, hidden_dim=64)
        
        # --- 2. Load Roots Internally ---
        # TODO: Update this path to match the repository structure
        roots_dir = 'C:/ORES_PINN/data/roots' # replace with the correct path
        
        def load_root_file(eta):
            path = os.path.join(roots_dir, f'eta={int(eta)}.npy')
            if not os.path.exists(path):
                raise FileNotFoundError(f"[KLE_PINN] Root file missing: {path}. Run generation script first.")
            return np.load(path)

        roots_30 = load_root_file(30.0)
        roots_40 = load_root_file(40.0)
        roots_50 = load_root_file(50.0)
        
        # --- 3. Parameter Fields (KLE Representations) ---
        nkl_lnKs = 14
        nkl_lnalpha = 11
        nkl_lnn = 9
        
        # Field 1: ln(Ks) | eta=30, nkl=14
        self.kle_ln_Ks = KLEField(
            mean=3.2173, variance=0.5, eta=30.0, 
            roots=roots_30[:nkl_lnKs], L=self.L, device=device
        )
        
        # Field 2: ln(alpha) | eta=40, nkl=11
        self.kle_ln_alpha = KLEField(
            mean=-3.3242, variance=0.3, eta=40.0, 
            roots=roots_40[:nkl_lnalpha], L=self.L, device=device
        )
        
        # Field 3: ln(n) | eta=50, nkl=9
        self.kle_ln_n = KLEField(
            mean=0.4447, variance=0.1, eta=50.0, 
            roots=roots_50[:nkl_lnn], L=self.L, device=device
        )
        
        # --- 4. Fixed VGM Constants ---
        self.theta_r = 0.078
        self.theta_s = 0.43

    def forward(self, t, z):
        """
        Forward pass to compute states and reconstruct parameters.
        
        Args:
            t (torch.Tensor): Time inputs (N, 1).
            z (torch.Tensor): Depth inputs (N, 1).
            
        Returns:
            Tuple containing (psi, theta, K, alpha, n, Ks).
        """
        
        # --- A. Predict State (Psi) using Neural Network ---
        t_norm, z_norm = normalize_tz(t, z)
        state_input = torch.cat([t_norm, z_norm], dim=-1)
        
        enc1_out = self.encoder_1(state_input)
        enc2_out = self.encoder_2(state_input)
        out_mlp = self.state_net(state_input, enc1_out, enc2_out)
        
        # Enforce negative pressure
        psi = -torch.exp(out_mlp)
        
        # --- B. Reconstruct Parameters using KLE ---
        # Note: Input to KLE is the physical z (unnormalized spatial coordinate)
        ln_Ks = self.kle_ln_Ks(z)
        ln_alpha = self.kle_ln_alpha(z)
        ln_n = self.kle_ln_n(z)
        
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