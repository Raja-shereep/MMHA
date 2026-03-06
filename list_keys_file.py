import torch
import sys

ckpt_path = r"C:\Users\sange\OneDrive\Desktop\C592 - MMHA-Net\model\alzheimer_cnn_best.pth"

try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        state_dict = ckpt['model_state']
        with open("model_keys.txt", "w") as f:
            for k, v in state_dict.items():
                f.write(f"{k}: {v.shape}\n")
    else:
        with open("model_keys.txt", "w") as f:
            f.write(f"No model_state found. Keys: {list(ckpt.keys())}\n")
except Exception as e:
    with open("model_keys.txt", "w") as f:
        f.write(f"Error: {e}\n")
