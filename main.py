from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import torch
import os
import uvicorn
import shutil
from typing import List
import json
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils.model_loader import load_mri_model, load_ecg_cnn_model, predict_mri, predict_ecg_cnn, MAPPING
from utils.gemini_utils import extract_eeg_features, analyze_clinical_text, generate_fusion_report
from utils.clinical_utils import load_clinical_model, predict_clinical, preprocess_clinical_data

app = FastAPI(title="MMHA-Net API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Static Files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Globals for models
mri_model = None
eeg_model = None
clinical_model = None
class_names = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check recursively or specifically
MRI_MODEL_PATH = os.path.join(BASE_DIR, "model", "alzheimer_cnn_best.pth")
EEG_MODEL_PATH = os.path.join(BASE_DIR, "model", "ecg_cnn_best.pth")
CLINICAL_MODEL_PATH = os.path.join(BASE_DIR, "model", "Clinical.pth")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    global mri_model, eeg_model, clinical_model, class_names
    
    # Load MRI Model
    if os.path.exists(MRI_MODEL_PATH):
        try:
            logger.info(f"Loading MRI Model from {MRI_MODEL_PATH}")
            mri_model, class_names = load_mri_model(MRI_MODEL_PATH, device)
            logger.info("MRI Model Loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load MRI model: {e}")
    else:
        logger.error(f"MRI Model not found at {MRI_MODEL_PATH}")

    # Load EEG (ECG CNN) Model
    if os.path.exists(EEG_MODEL_PATH):
        try:
            logger.info(f"Loading EEG (ECG CNN) Model from {EEG_MODEL_PATH}")
            # The model is an ECG CNN now, replacing the simple ANN
            eeg_model = load_ecg_cnn_model(EEG_MODEL_PATH, device=device)
            logger.info("EEG (ECG) Model Loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load EEG model: {e}")
    else:
        logger.error(f"EEG Model not found at {EEG_MODEL_PATH}")

    # Load Clinical Model
    if os.path.exists(CLINICAL_MODEL_PATH):
        try:
             logger.info(f"Loading Clinical Model from {CLINICAL_MODEL_PATH}")
             clinical_model = load_clinical_model(CLINICAL_MODEL_PATH, device)
        except Exception as e:
            logger.error(f"Failed to load Clinical model: {e}")
    else:
        logger.error(f"Clinical Model not found at {CLINICAL_MODEL_PATH}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r") as f:
            return f.read()
    return "<h1>MMHA-Net API is running. Frontend not found.</h1>"

@app.post("/predict_mri")
async def predict_mri_endpoint(file: UploadFile = File(...)):
    if mri_model is None:
        raise HTTPException(status_code=503, detail="MRI Model not loaded")
    
    temp_path = os.path.join(TEMP_DIR, file.filename)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = predict_mri(mri_model, temp_path, class_names, device)
        
        # Heuristic Disease Progression Logic based on confidence
        confidence = result['confidence']
        stage_risk = "Stable"
        mapped = result['mapped_class']
        
        if confidence < 0.6:
            stage_risk = "Uncertain / Monitoring Required"
        elif mapped == "CN":
             stage_risk = "Low Risk"
        elif mapped == "MCI":
             stage_risk = "High Risk of Progression to AD" if confidence > 0.8 else "Moderate Risk"
        elif mapped == "AD":
             stage_risk = "Advanced Stage"
             
        result['disease_progression'] = stage_risk
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.post("/predict_eeg")
async def predict_eeg_endpoint(file: UploadFile = File(...)):
    if eeg_model is None:
        raise HTTPException(status_code=503, detail="EEG Model not loaded")
    
    temp_path = os.path.join(TEMP_DIR, f"eeg_{file.filename}")
    try:
        # Save uploaded image
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Class names as provided by user:
        # ['abnormal_heartbeat_ecg_images', 'myocardial_infarction_ecg_images', 'normal_ecg_images', 'post_mi_history_ecg_images']
        ecg_class_names = [
            'abnormal_heartbeat_ecg_images', 
            'myocardial_infarction_ecg_images', 
            'normal_ecg_images', 
            'post_mi_history_ecg_images'
        ]

        logger.info("Running ECG/EEG CNN prediction...")
        result = predict_ecg_cnn(eeg_model, temp_path, ecg_class_names, device)
        
        return result

    except Exception as e:
        logger.error(f"EEG Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
         if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.post("/predict_clinical")
async def predict_clinical_endpoint(data: dict):
    if clinical_model is None:
        raise HTTPException(status_code=503, detail="Clinical Model not loaded")

    try:
        logger.info(f"Received clinical data: {data}")
        # Preprocess
        features = preprocess_clinical_data(data)
        
        # Predict
        result = predict_clinical(clinical_model, features, device)
        
        return result
        
    except Exception as e:
        logger.error(f"Clinical prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


from typing import Optional

class FusionRequest(BaseModel):
    mri_result: Optional[dict] = None
    eeg_result: Optional[dict] = None
    clinical_result: Optional[dict] = None

@app.post("/predict_fusion")
async def predict_fusion_endpoint(request: FusionRequest):
    try:
        logger.info("Generating multimodal fusion report...")
        report_html = generate_fusion_report(
            request.mri_result, 
            request.eeg_result, 
            request.clinical_result
        )
        return {"report_html": report_html}
    except Exception as e:
        logger.error(f"Fusion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
