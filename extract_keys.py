import torch
import sys

ckpt_path = r"C:\Users\sange\OneDrive\Desktop\C592 - MMHA-Net\model\alzheimer_cnn_best.pth"

try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    with open("model_structure.txt", "w") as f:
        f.write(f"Keys: {list(ckpt.keys())}\n")
        if 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
            f.write(f"State dict keys (first 50):\n")
            for i, k in enumerate(sd.keys()):
                if i > 50: break
                f.write(f"{k}: {sd[k].shape}\n")
        else:
            f.write("No model_state_dict found in keys.\n")
            # Maybe it IS the state dict?
            if isinstance(ckpt, dict):
                 f.write("Checking if ckpt itself is state dict:\n")
                 for i, k in enumerate(ckpt.keys()):
                    if i > 20: break
                    f.write(f"{k}: {ckpt[k]}\n")

except Exception as e:
    with open("model_structure.txt", "w") as f:
        f.write(f"Error: {e}")
