import torch
import torch.nn as nn
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

# --- Model Definition ---
class ClinicalModel(nn.Module):
    def __init__(self, input_size=32):
        super(ClinicalModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# --- Loading Function ---
def load_clinical_model(ckpt_path: str, device):
    try:
        logger.info(f"Loading Clinical model from {ckpt_path}...")
        # Initialize model with 32 inputs as per inspection
        model = ClinicalModel(input_size=32).to(device)
        
        # Load state dict
        if torch.cuda.is_available():
            state_dict = torch.load(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
            
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("Clinical Model Loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load Clinical model: {e}")
        return None

# --- Preprocessing Function ---
def preprocess_clinical_data(data: dict) -> np.ndarray:
    """
    Converts the 15 user-provided fields into the 32-feature vector expected by the model.
    
    Mapping Strategy (Best Effort):
    1. Numerical Scaling: Age, BMI, MMSE, ADL are scaled (min-max or standard).
       *NOTE*: Without the original scaler, we use estimated bounds.
    2. Categorical Encoding: Gender, Ethnicity, Education, Smoking, etc. are One-Hot Encoded.
    3. Padding: Remaining slots are filled with 0.
    """
    
    # 1. Initialize 32-float vector
    features = np.zeros(32, dtype=np.float32)
    
    try:
        # --- Numerical Features (Indices 0-3) ---
        # Age: Assume range 50-90
        age = float(data.get('Age', 70))
        features[0] = (age - 50) / (90 - 50)
        
        # BMI: Assume range 15-40
        bmi = float(data.get('BMI', 25))
        features[1] = (bmi - 15) / (40 - 15)
        
        # MMSE: Range 0-30 (30 is normal, <24 is abnormal)
        mmse = float(data.get('MMSE', 30))
        features[2] = mmse / 30.0
        
        # ADL: Range 0-10 (10 is independent)
        adl = float(data.get('ADL', 10))
        features[3] = adl / 10.0
        
        # --- binary/Categorical Features (Indices 4-31) ---
        
        # Gender (Index 4)
        features[4] = 1.0 if data.get('Gender') == 'Male' else 0.0
        
        # Ethnicity (Indices 5-8: Caucasian, AfricanAmerican, Asian, Hispanic)
        ethnicity = data.get('Ethnicity', 'Caucasian')
        if ethnicity == 'Caucasian': features[5] = 1.0
        elif ethnicity == 'African American': features[6] = 1.0
        elif ethnicity == 'Asian': features[7] = 1.0
        elif ethnicity == 'Hispanic': features[8] = 1.0
        
        # Education (Indices 9-12: None, HighSchool, Bachelors, Higher)
        edu = data.get('EducationLevel', 'HighSchool')
        if edu == 'None': features[9] = 1.0
        elif edu == 'High School': features[10] = 1.0
        elif edu == 'Bachelor\'s': features[11] = 1.0
        elif edu == 'Higher': features[12] = 1.0
        
        # Smoking (Indices 13-15: Never, Former, Current)
        smoke = data.get('Smoking', 'Never')
        if smoke == 'Never': features[13] = 1.0
        elif smoke == 'Former': features[14] = 1.0
        elif smoke == 'Current': features[15] = 1.0
        
        # Health Conditions (Binary 0/1)
        features[16] = 1.0 if data.get('FamilyHistoryAlzheimers') == 'Yes' else 0.0
        features[17] = 1.0 if data.get('Diabetes') == 'Yes' else 0.0
        features[18] = 1.0 if data.get('Hypertension') == 'Yes' else 0.0
        
        # Symptoms (Binary 0/1)
        features[19] = 1.0 if data.get('MemoryComplaints') == 'Yes' else 0.0
        features[20] = 1.0 if data.get('Confusion') == 'Yes' else 0.0
        features[21] = 1.0 if data.get('Disorientation') == 'Yes' else 0.0
        features[22] = 1.0 if data.get('Forgetfulness') == 'Yes' else 0.0
        
        # Padding indices 23-31 are left as 0.0
        
        logger.debug(f"Preprocessed features: {features}")
        return features
        
    except Exception as e:
        logger.error(f"Error in preprocessing clinical data: {e}")
        # Return zeros on error to prevent crash, but log heavily
        return np.zeros(32, dtype=np.float32)

# --- Prediction Function ---
def predict_clinical(model, features: np.ndarray, device):
    """
    Predicts Alzheimer's risk using the preprocessed 32-feature vector.
    """
    try:
        tensor = torch.tensor(features).unsqueeze(0).to(device) # Shape [1, 32]
        
        with torch.no_grad():
            output = model(tensor)
            prob = output.item()
            
        # Interpretation
        label = "High Risk (AD)" if prob > 0.5 else "Low Risk (CN)"
        
        return {
            "probability": prob,
            "label": label,
            "risk_level": "High" if prob > 0.7 else "Moderate" if prob > 0.4 else "Low",
             # Adding summary/keywords to match the format expected by fusion
            "summary": f"Clinical model predicts {label} with probability {prob:.2f}",
            "keywords": ["Clinical Model", label]
        }
        
    except Exception as e:
        logger.error(f"Error in clinical prediction: {e}")
        return {"error": str(e)}
