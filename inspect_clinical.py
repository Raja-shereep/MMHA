import torch
import os

model_path = r"c:\Users\sange\OneDrive\Desktop\C592 - MMHA-Net\model\Clinical.pth"

try:
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        print("Keys in state_dict:", state_dict.keys())
        for key, value in state_dict.items():
            print(f"{key}: {value.shape}")
    else:
        print(f"Model file not found at {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
