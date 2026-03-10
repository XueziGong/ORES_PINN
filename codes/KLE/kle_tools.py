import numpy as np
import torch
from scipy.optimize import brentq

def characteristic_equation(w, eta, L):
    """
    Characteristic equation for the exponential covariance kernel.
    Equation: (eta^2 * w^2 - 1) * sin(w * L) - 2 * eta * w * cos(w * L) = 0
    
    Args:
        w (float or np.array): Frequency/Eigenvalue parameter (omega).
        eta (float): Correlation length.
        L (float): Domain length.
        
    Returns:
        float or np.array: The value of the characteristic function.
    """
    return (eta**2 * w**2 - 1) * np.sin(w * L) - 2 * eta * w * np.cos(w * L)

def find_roots(eta, L, n_roots, w_max=20, step=0.001):
    """
    Find positive roots of the characteristic equation using Brent's method.
    
    Args:
        eta (float): Correlation length.
        L (float): Domain length.
        n_roots (int): Number of roots to find.
        w_max (float): Maximum range to search for roots.
        step (float): Step size for the initial coarse search.
        
    Returns:
        np.array: Array of the first n_roots positive roots.
    """
    w_values = np.arange(0, w_max, step)
    roots = []

    def _func(w):
        return characteristic_equation(w, eta, L)

    for i in range(len(w_values) - 1):
        w1, w2 = w_values[i], w_values[i + 1]
        # Check for sign change indicating a root lies between w1 and w2
        if _func(w1) * _func(w2) < 0:
            try:
                # Refine the root using Brent's method
                root = brentq(_func, w1, w2)
                # Filter out 0 and roots that are too close to each other
                if root > 1e-6 and (not roots or np.abs(root - roots[-1]) > 1e-4):
                    roots.append(root)
                if len(roots) >= n_roots:
                    break
            except ValueError:
                continue
                
    return np.array(roots)

def compute_eigenvalues(eta, variance, wn):
    """
    Calculate eigenvalues lambda_n based on the roots wn.
    Formula: lambda_n = (2 * eta * variance) / (eta^2 * wn^2 + 1)
    
    Args:
        eta (float): Correlation length.
        variance (float): Variance of the field (sigma^2).
        wn (float, np.array, or torch.Tensor): The roots (omega_n).
        
    Returns:
        The computed eigenvalues.
    """
    if isinstance(wn, torch.Tensor):
        return (2 * eta * variance) / (eta**2 * wn**2 + 1)
    else:
        return (2 * eta * variance) / (eta**2 * wn**2 + 1)

def compute_eigenfunctions(eta, wn, z, L):
    """
    Calculate eigenfunctions phi_n(z).
    Formula: phi_n(z) = factor * (eta * wn * cos(wn * z) + sin(wn * z))
    
    Note:
    This implementation assumes a positive coordinate system z in [0, L].
    z=0 corresponds to the bottom, z=L corresponds to the top/surface.
    No coordinate mapping (like abs) is needed if inputs are positive.
    
    Args:
        eta (float): Correlation length.
        wn (float, np.array, or torch.Tensor): The roots (omega_n).
        z (float, np.array, or torch.Tensor): Coordinate values.
        L (float): Domain length.
        
    Returns:
        The computed eigenfunctions.
    """
    # Select appropriate math library (numpy or torch)
    if isinstance(wn, torch.Tensor) or isinstance(z, torch.Tensor):
        sin_func = torch.sin
        cos_func = torch.cos
        sqrt_func = torch.sqrt
    else:
        sin_func = np.sin
        cos_func = np.cos
        sqrt_func = np.sqrt

    # Normalization factor
    denominator = (eta**2 * wn**2 + 1) * L / 2.0 + eta
    factor = 1.0 / sqrt_func(denominator)
    
    # Eigenfunction calculation
    return factor * (eta * wn * cos_func(wn * z) + sin_func(wn * z))