import torch
import sys

ckpt_path = r"C:\Users\sange\OneDrive\Desktop\C592 - MMHA-Net\model\alzheimer_cnn_best.pth"

try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print("Keys in checkpoint:", ckpt.keys())
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        print("First 20 keys in state_dict:")
        for i, key in enumerate(state_dict.keys()):
            if i >= 20: break
            print(key, state_dict[key].shape)
    else:
        print("No model_state_dict found.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
