import torch
import sys
import os

ckpt_path = r"C:\Users\sange\OneDrive\Desktop\C592 - MMHA-Net\model\ecg_cnn_best.pth"

try:
    if not os.path.exists(ckpt_path):
        print(f"File not found: {ckpt_path}")
        sys.exit(1)
        
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # It's a dict passed directly
    state_dict = ckpt
    
    print("--- Shapes ---")
    keys_to_check = [
        'features.0.0.weight', 
        'features.3.0.weight', 
        'classifier.1.weight',
        'classifier.5.weight',
        'classifier.9.weight'
    ]
    
    for k in keys_to_check:
        if k in state_dict:
            print(f"{k}: {state_dict[k].shape}")
        else:
            print(f"{k}: Not Found")

except Exception as e:
    print(f"Error loading checkpoint: {e}")
