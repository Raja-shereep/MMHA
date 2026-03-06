import requests
import os
from PIL import Image

# Create a dummy image
img_path = "temp_test_ecg.jpg"
Image.new('RGB', (224, 224), color='red').save(img_path)

url = "http://127.0.0.1:8000/predict_eeg"
files = {'file': open(img_path, 'rb')}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(response.json())
except Exception as e:
    print(f"Error: {e}")
finally:
    if os.path.exists(img_path):
        files['file'].close()
        os.remove(img_path)
