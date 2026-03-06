
import requests
import json
import os

BASE_URL = "http://127.0.0.1:8000"

def test_eeg():
    print("\nTesting EEG Endpoint...")
    # Create a dummy image
    with open("test_eeg.txt", "w") as f:
        f.write("Dummy EEG Image Content")
    
    files = {'file': ('test_eeg.txt', open('test_eeg.txt', 'rb'), 'text/plain')}
    # Note: In real app this should be an image, but our extract_eeg_features might fail gracefully or return dummy features
    # depending on implementation. gemini might error on text file sent as image.
    # Let's see if we can use a dummy image if possible, or just expect the error handling to work.
    
    try:
        response = requests.post(f"{BASE_URL}/predict_eeg", files=files)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        files['file'][1].close()
        os.remove("test_eeg.txt")

def test_clinical():
    print("\nTesting Clinical Endpoint...")
    with open("test_clinical.txt", "w") as f:
        f.write("Patient is a 70 year old male showing signs of memory loss and confusion.")
        
    files = {'file': ('test_clinical.txt', open('test_clinical.txt', 'rb'), 'text/plain')}
    
    try:
        response = requests.post(f"{BASE_URL}/predict_clinical", files=files)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        files['file'][1].close()
        os.remove("test_clinical.txt")

def test_fusion():
    print("\nTesting Fusion Endpoint...")
    payload = {
        "mri_result": {"mapped_class": "AD", "confidence": 0.95, "disease_progression": "High Risk"},
        "eeg_result": {"label": "Positive", "probability": 0.88, "features": {"alpha": 10}},
        "clinical_result": {"risk_level": "High", "summary": "Memory loss detected", "keywords": ["memory", "loss"]}
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict_fusion", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_eeg()
    test_clinical()
    test_fusion()
