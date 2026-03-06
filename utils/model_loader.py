import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# --- Model Definitions ---

class AlzheimerCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(AlzheimerCNN, self).__init__()
        self.features = nn.Sequential(
            self._make_block(3, 32),
            self._make_block(32, 64),
            self._make_block(64, 128),
            self._make_block(128, 256)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

class SimpleANN(nn.Module):
    def __init__(self, input_size):
        super(SimpleANN, self).__init__()
        self.layer_1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 1) # Output layer for binary classification

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.layer_3(x)
        return x

# --- Loading Functions ---

# --- Loading Functions ---

def load_mri_model(ckpt_path: str, device):
    print(f"Loading MRI model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Handle keys: 'model_state' vs 'model_state_dict'
    state_dict = ckpt.get('model_state', ckpt.get('model_state_dict'))
    if state_dict is None:
        raise ValueError("Could not find 'model_state' or 'model_state_dict' in checkpoint")

    class_names = ckpt.get('class_names', ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'])
    num_classes = len(class_names)
    
    model = AlzheimerCNN(num_classes=num_classes).to(device)
    # strict=False to allow for minor discrepancies (e.g. running_mean/var tracking)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, class_names

def load_eeg_model(ckpt_path: str, input_size: int, device):
    # Backward compatibility if needed, but we are moving to CNN
    print(f"Loading EEG model (SimpleANN) from {ckpt_path}...")
    model = SimpleANN(input_size).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model

# --- ECG/EEG CNN Model ---

class ECGCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(ECGCNN, self).__init__()
        self.features = nn.Sequential(
            self._make_block(3, 32),
            self._make_block(32, 64),
            self._make_block(64, 128),
            self._make_block(128, 256)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 1024), # 50176 -> 1024
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

def load_ecg_cnn_model(ckpt_path: str, device):
    print(f"Loading ECG CNN model from {ckpt_path}...")
    # Dictionary is loaded directly
    state_dict = torch.load(ckpt_path, map_location=device)
    
    # Infer num_classes from last layer weight
    # classifier.9.weight => [num_classes, 512]
    num_classes = 4
    if 'classifier.9.weight' in state_dict:
        num_classes = state_dict['classifier.9.weight'].shape[0]
        
    model = ECGCNN(num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- Prediction Functions ---

MAPPING = {
    "NonDemented": "CN",
    "VeryMildDemented": "MCI",
    "MildDemented": "MCI",
    "ModerateDemented": "AD"
}

ECG_MAPPING = {
    'abnormal_heartbeat_ecg_images': 'Abnormal Heartbeat',
    'myocardial_infarction_ecg_images': 'Myocardial Infarction',
    'normal_ecg_images': 'Normal',
    'post_mi_history_ecg_images': 'Post MI History'
}

def predict_mri(model, image_path: str, class_names: list, device):
    img_size = 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = probs.argmax()
        
    predicted_class_raw = class_names[pred_idx]
    predicted_class_mapped = MAPPING.get(predicted_class_raw, predicted_class_raw)
    
    return {
        "raw_class": predicted_class_raw,
        "mapped_class": predicted_class_mapped,
        "confidence": float(probs[pred_idx]),
        "all_probabilities": dict(zip(class_names, probs.tolist()))
    }

def predict_eeg(model, features: list, device, scaler=None):
    # Legacy function for SimpleANN
    features_array = np.array([features], dtype=np.float32)
    inputs = torch.tensor(features_array).to(device)
    
    with torch.no_grad():
        output = model(inputs)
        prob = torch.sigmoid(output).item()
        
    label = "Positive" if prob > 0.5 else "Negative"
    
    return {
        "probability": prob,
        "label": label
    }

def predict_ecg_cnn(model, image_path: str, class_names: list, device):
    # Transformation as per user request (User code snippet implies standard eval_transforms)
    # We will use standard Resize/Normalize similar to MRI for now, assuming 224x224 training
    img_size = 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], # ImageNet stats usually
                             [0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx = probs.argmax()
    pred_class = class_names[pred_idx]
    mapped_class = ECG_MAPPING.get(pred_class, pred_class)
    
    return {
        "predicted_class": pred_class,
        "mapped_class": mapped_class,
        "confidence": float(probs[pred_idx]),
        "all_probabilities": dict(zip(class_names, probs.tolist()))
    }

