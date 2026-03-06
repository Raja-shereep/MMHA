import torch

ckpt_path = r"C:\Users\sange\OneDrive\Desktop\C592 - MMHA-Net\model\alzheimer_cnn_best.pth"

try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        state_dict = ckpt['model_state']
        print("Keys and Shapes:")
        for k, v in state_dict.items():
            print(f"{k}: {v.shape}")
    else:
        print("No model_state found. Keys:", ckpt.keys())
except Exception as e:
    print(f"Error: {e}")
