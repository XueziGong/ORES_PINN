import torch
import torch.nn as nn
import numpy as np

# ==========================================
# 1. Helper Functions & Layers
# ==========================================

def initialize_weights(model):
    """
    Xavier initialization for network weights.
    """
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

def normalize_tz(t, z):
    """
    Normalize spatial-temporal inputs to the range [-1, 1].
    Assumes t in [0, 10] and z in [-99, 0].
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
        self.fc5 = nn.Linear(hidden_dim, 1)  # Output dim is 1 (psi)
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
    """
    Standard MLP architecture.
    Structure: [input_dim, hidden, hidden, hidden, hidden, output_dim]
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
        initialize_weights(self)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. Main PINN Model Class
# ==========================================

class PINN(nn.Module):
    def __init__(self, device):
        super(PINN, self).__init__()
        self.device = device
        
        # State Network (Predicts Psi)
        # Inputs: t, z (2D) | Hidden: 64 | Output: 1D
        self.encoder_1 = Encoder(2, 64)
        self.encoder_2 = Encoder(2, 64)
        self.state_net = ModifiedMLP(2, hidden_dim=64)
        
        # Parameter Networks (Predicts Heterogeneity)
        # Three separate MLPs predict fluctuations for alpha, n, and Ks
        # Input: z (1D) | Hidden: 36 | Output: 1D
        self.net_ln_alpha = StandardMLP(input_dim=1, output_dim=1)
        self.net_ln_n = StandardMLP(input_dim=1, output_dim=1)
        self.net_ln_Ks = StandardMLP(input_dim=1, output_dim=1)
        
        # Fixed VGM Parameters (Assumed constant for this soil type context)
        self.theta_r = 0.078
        self.theta_s = 0.43
        
        # Parameter Means (Priors)
        self.register_buffer('mu_lnKs', torch.tensor(3.2173))
        self.register_buffer('mu_lnalpha', torch.tensor(-3.3242))
        self.register_buffer('mu_lnn', torch.tensor(0.4447))

    def forward(self, t, z):
        """
        Forward pass to compute all states and parameters.
        
        Args:
            t (torch.Tensor): Time inputs (N, 1)
            z (torch.Tensor): Depth inputs (N, 1)
            
        Returns:
            psi (torch.Tensor): Predicted pressure head
            theta (torch.Tensor): Calculated water content via VGM
            K (torch.Tensor): Calculated hydraulic conductivity via VGM
            alpha (torch.Tensor): Predicted alpha parameter
            n (torch.Tensor): Predicted n parameter
            Ks (torch.Tensor): Predicted Ks parameter
        """
        
        # 1. Normalize Inputs
        t_norm, z_norm = normalize_tz(t, z)
        state_input = torch.cat([t_norm, z_norm], dim=-1)  # (N, 2)
        param_input = z_norm  # (N, 1) - Parameters depend only on space
        
        # 2. Predict State (Psi)
        enc1_out = self.encoder_1(state_input)
        enc2_out = self.encoder_2(state_input)
        out_mlp = self.state_net(state_input, enc1_out, enc2_out)
        
        # Enforce negative pressure head assumption: psi = -exp(NN_out)
        psi = -torch.exp(out_mlp)
        
        # 3. Predict Parameters (Alpha, n, Ks) separately
        # Outputs represent the fluctuation from the spatial mean
        d_ln_alpha = self.net_ln_alpha(param_input)  # (N, 1)
        d_ln_n = self.net_ln_n(param_input)          # (N, 1)
        d_ln_Ks = self.net_ln_Ks(param_input)        # (N, 1)
        
        # Reconstruct physical parameters in Log scale then Exp
        ln_alpha = self.mu_lnalpha + d_ln_alpha
        ln_n = self.mu_lnn + d_ln_n
        ln_Ks = self.mu_lnKs + d_ln_Ks
        
        alpha = torch.exp(ln_alpha)
        n = torch.exp(ln_n)
        Ks = torch.exp(ln_Ks)
        
        # 4. Compute Derived Variables (Theta, K) using VGM Model
        m = 1.0 - (1.0 / n)
        abs_psi = -psi
        
        # Effective Saturation (Se)
        se_term = 1.0 + (alpha * abs_psi).pow(n)
        Se = se_term.pow(-m)
        
        # Water Content (Theta)
        theta = self.theta_r + Se * (self.theta_s - self.theta_r)
        
        # Hydraulic Conductivity (K)
        term1 = Se.pow(0.5)
        se_inv_m = 1.0 / se_term
        inner_term = torch.relu(1.0 - se_inv_m)
        term2 = (1.0 - inner_term.pow(m)).pow(2)
        
        K = Ks * term1 * term2
        
        return psi, theta, K, alpha, n, Ks