import torch
import numpy as np
import importlib
from unittest.mock import patch
import os

# Define the model filenames to evaluate (without .py extension)
MODEL_FILES = ["PINN", "PINN_SUB", "KLE_PINN", "RES_PINN", "ORES_PINN"]

# Mock numpy.load to instantiate models without requiring actual .npy data files
def mock_load_root_file(*args, **kwargs):
    # Generate a sufficiently large dummy array to prevent out-of-bounds slicing
    return np.zeros(100)

def mock_exists(*args, **kwargs):
    return True

@patch('numpy.load', side_effect=mock_load_root_file)
@patch('os.path.exists', side_effect=mock_exists)
def count_all_models(mock_exists, mock_load):
    device = torch.device('cpu')
    
    # Define prefixes for State (Psi) and Parameter network modules
    state_prefixes = ('encoder_1', 'encoder_2', 'state_net')
    param_prefixes = (
        'param_net', 'net_ln_alpha', 'net_ln_n', 'net_ln_Ks', 
        'kle_ln_Ks', 'kle_ln_alpha', 'kle_ln_n', 
        'gamma_logits_Ks', 'gamma_logits_alpha', 'gamma_logits_n'
    )

    print("="*50)
    print(" Trainable Parameters Count for PINN Models")
    print("="*50)

    for mod_name in MODEL_FILES:
        try:
            # Dynamically import the model module
            mod = importlib.import_module(mod_name)
            
            # Instantiate the PINN model
            model = mod.PINN(device=device)
            
            state_params = 0
            param_params = 0
            unclassified_params = 0
            
            # Iterate through all parameters and classify them by name
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                num_params = param.numel()
                
                if name.startswith(state_prefixes):
                    state_params += num_params
                elif name.startswith(param_prefixes):
                    param_params += num_params
                else:
                    unclassified_params += num_params
                    print(f"  [Warning] Unclassified parameter: {name} ({num_params})")
            
            total_params = state_params + param_params + unclassified_params
            
            print(f"Model: {mod_name}.py")
            print(f"  ├─ State (Psi) Network: {state_params:,} parameters")
            print(f"  ├─ Parameter Network  : {param_params:,} parameters")
            print(f"  └─ Total Trainable    : {total_params:,} parameters")
            print("-" * 50)
            
        except ImportError:
            print(f"[Error] Module not found: {mod_name}.py. Ensure it is in the same directory.")
        except Exception as e:
            print(f"[Error] Failed to parse {mod_name}: {e}")

if __name__ == "__main__":
    count_all_models()