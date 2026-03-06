import requests

url = "http://127.0.0.1:8001/predict_eeg"
data = {
    "features": "0,0,0,0,0,0,0,0,0,0"
}

try:
    response = requests.post(url, data=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.json()}")
    
    if response.status_code == 200:
        print("SUCCESS: Endpoint accepted comma-separated values.")
    else:
        print("FAILURE: Endpoint returned error.")

except Exception as e:
    print(f"Error connecting to server: {e}")
