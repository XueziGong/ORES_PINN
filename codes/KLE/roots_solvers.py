import numpy as np
import matplotlib.pyplot as plt
import os
from kle_tools import find_roots, characteristic_equation

# --- Configuration Parameters ---
L = 99.0      # Domain length (z from 0 to 99)
eta = 50.0    # Correlation length
nkl = 9      # Number of KL expansion terms to keep

# --- Directories (Updated) ---
# Absolute path for storing roots
save_dir = r'...\data\roots' # replace with the correct path

if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
    except OSError as e:
        print(f"Error creating directory {save_dir}: {e}")
        exit()

print(f"Solving roots for L={L}, eta={eta}, nkl={nkl}...")

# --- 1. Solve for Roots ---
# We search up to w=10.0, which is usually sufficient for the first few roots
roots = find_roots(eta, L, nkl, w_max=10.0)

# Save the results
# Modified: Filename is now just eta=xx.npy
save_name = f'eta={int(eta)}.npy'
save_path = os.path.join(save_dir, save_name)
np.save(save_path, roots)

print(f"Roots found: {roots}")
print(f"Saved to: {save_path}")

# --- 2. Visualization of Roots ---
# Create a dense grid of w values to plot the characteristic equation curve
w_plot = np.linspace(0, roots[-1] + 0.5, 1000)
eq_values = characteristic_equation(w_plot, eta, L)

plt.figure(figsize=(10, 6))
plt.plot(w_plot, eq_values, label='Characteristic Equation', color='blue')
plt.axhline(0, color='black', linewidth=1, linestyle='--') # Zero line

# Mark the found roots
plt.scatter(roots, np.zeros_like(roots), color='red', zorder=5, label='Found Roots')
for i, root in enumerate(roots):
    plt.annotate(f'$\omega_{i+1}$', (root, 0.1), xytext=(root, 0.5), 
                 arrowprops=dict(arrowstyle='->', color='red'))

plt.title(f'Characteristic Equation & Roots (L={L}, $\eta$={eta})')
plt.xlabel('$\omega$ (Frequency)')
plt.ylabel('Equation Value')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the plot
# Also updated the plot filename to include eta, so different runs don't overwrite each other
plot_path = os.path.join(save_dir, f'roots_visualization_eta={int(eta)}.png')
plt.savefig(plot_path, dpi=300)
print(f"Roots visualization saved to: {plot_path}")
plt.show()