import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_manual_clinical():
    print("\nTesting Manual Clinical Endpoint...")
    
    # 15 Features as per user request
    payload = {
        "Age": 75,
        "Gender": "Male",
        "Ethnicity": "Caucasian",
        "EducationLevel": "Bachelor's",
        "BMI": 28.5,
        "Smoking": "Former",
        "MMSE": 22,
        "ADL": 6,
        "FamilyHistoryAlzheimers": "Yes",
        "Diabetes": "Yes",
        "Hypertension": "Yes",
        "MemoryComplaints": "Yes",
        "Confusion": "Yes",
        "Disorientation": "No",
        "Forgetfulness": "Yes"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict_clinical", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("SUCCESS: Clinical manual entry endpoint works.")
        else:
            print("FAILURE: Endpoint returned error.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_manual_clinical()
