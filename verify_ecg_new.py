import torch
from utils.model_loader import load_ecg_cnn_model, predict_ecg_cnn
import os
from PIL import Image

# Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "ecg_cnn_best.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    try:
        model = load_ecg_cnn_model(MODEL_PATH, DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create dummy image
    dummy_img_path = "dummy_ecg.jpg"
    img = Image.new('RGB', (224, 224), color='red')
    img.save(dummy_img_path)

    classes = [
            'abnormal_heartbeat_ecg_images', 
            'myocardial_infarction_ecg_images', 
            'normal_ecg_images', 
            'post_mi_history_ecg_images'
        ]

    try:
        result = predict_ecg_cnn(model, dummy_img_path, classes, DEVICE)
        print("Prediction result:", result)
    except Exception as e:
        print(f"Prediction failed: {e}")
    finally:
        if os.path.exists(dummy_img_path):
            os.remove(dummy_img_path)

if __name__ == "__main__":
    test_model()
