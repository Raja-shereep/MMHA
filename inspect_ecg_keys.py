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
    
    print("--- All Keys ---")
    for k in state_dict.keys():
        print(k)

except Exception as e:
    print(f"Error loading checkpoint: {e}")
