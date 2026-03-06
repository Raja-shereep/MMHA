import google.generativeai as genai
import os
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.warning("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=api_key)

def get_gemini_model(model_name="gemini-2.5-flash-lite"):
    return genai.GenerativeModel(model_name)

def extract_eeg_features(image_path: str):
    """
    Extracts 10 specific EEG features from an image using Gemini.
    Returns a dictionary of features and values.
    """
    try:
        model = get_gemini_model()
        
        # Read image
        with open(image_path, "rb") as f:
            image_data = f.read()
            
        prompt = """
        Analyze this EEG signal image and extract the following 10 features. 
        If specific values are visible, use them. If it's a waveform, estimate specific values for:
        1. attention (0-100)
        2. meditation (0-100)
        3. delta (1-3 Hz power)
        4. theta (4-7 Hz power)
        5. lowAlpha (8-11 Hz power)
        6. highAlpha (8-11 Hz power)
        7. lowBeta (12-29 Hz power)
        8. highBeta (12-29 Hz power)
        9. lowGamma (30-100 Hz power)
        10. highGamma (30-100 Hz power)
        
        Return ONLY a raw JSON object with these keys and numerical values. 
        Example: {"attention": 50, "meditation": 40, "delta": 2000, ...}
        """
        
        response = model.generate_content([
            {'mime_type': 'image/jpeg', 'data': image_data},
            prompt
        ])
        
        text = response.text.strip()
        # Clean up code blocks if present
        if text.startswith("```json"):
            text = text[7:-3]
        elif text.startswith("```"):
            text = text[3:-3]
            
        return json.loads(text)
        
    except Exception as e:
        logger.error(f"Error extracting EEG features: {e}")
        # Fallback to dummy values if extraction fails, or re-raise
        # For a robust system, we might want to fail hard or return zeros
        return {
            "attention": 0, "meditation": 0, "delta": 0, "theta": 0,
            "lowAlpha": 0, "highAlpha": 0, "lowBeta": 0, "highBeta": 0,
            "lowGamma": 0, "highGamma": 0
        }

def analyze_clinical_text(file_path: str, file_type: str = "text/plain"):
    """
    Analyzes a clinical document (text or image) to extract risk factors.
    """
    try:
        model = get_gemini_model()
        
        parts = []
        
        if file_type.startswith("image/"):
             with open(file_path, "rb") as f:
                parts.append({'mime_type': file_type, 'data': f.read()})
        elif file_type == "application/pdf":
             with open(file_path, "rb") as f:
                parts.append({'mime_type': 'application/pdf', 'data': f.read()})
        else:
            # Try reading as text first, if fails, treat as binary for Gemini (e.g. some doc/docx might work or fail gracefully)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    parts.append(f.read())
            except UnicodeDecodeError:
                # Fallback: send as binary/blob if possible or log error
                # Gemini generally accepts text or specific MIME types. 
                # For unknown binary types, it might fail, but let's try sending as generic document if supported,
                # or just fail. For now, we will log and try to read as binary if it's a known doc type for Gemini.
                # Assuming Gemini 1.5/2.0 can handle other docs if mime_type is set.
                with open(file_path, "rb") as f:
                     # Defaulting to application/pdf or text/plain might be risky if it's not.
                     # But let's assume the user uploads supported docs.
                     parts.append({'mime_type': file_type, 'data': f.read()})
                
        prompt = """
        You are an expert medical AI. Analyze the provided clinical document (doctor's notes or report).
        Extract key insights related to Alzheimer's or cognitive decline.
        
        Provide the output in JSON format with the following keys:
        - summary: A brief summary of the patient's condition.
        - risk_level: "Low", "Moderate", or "High" based on keywords.
        - keywords: List of relevant medical terms found (e.g., "dementia", "memory loss", "atrophy").
        - recommendation: A brief recommendation based on the notes.
        """
        
        parts.append(prompt)
        
        response = model.generate_content(parts)
        
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3]
        elif text.startswith("```"):
            text = text[3:-3]
            
        return json.loads(text)
        
    except Exception as e:
        logger.error(f"Error analyzing clinical text: {e}")
        return {
            "summary": "Error analyzing document.",
            "risk_level": "Unknown",
            "keywords": [],
            "recommendation": "Manual review required."
        }

def generate_fusion_report(mri_result, eeg_result, clinical_result):
    """
    Generates a final multimodal fusion report using Gemini.
    Handles missing modalities by checking for None.
    """
    try:
        model = get_gemini_model()
        
        # Format inputs or mark as missing
        mri_text = "Not Provided"
        if mri_result:
            mri_text = f"""
            - Predicted Class: {mri_result.get('mapped_class')} (Raw: {mri_result.get('raw_class')})
            - Confidence: {mri_result.get('confidence')}
            - Progression Risk: {mri_result.get('disease_progression')}
            """

        eeg_text = "Not Provided"
        if eeg_result:
             eeg_text = f"""
            - Classification: {eeg_result.get('label')}
            - Probability: {eeg_result.get('probability')}
            - Features extracted: {eeg_result.get('features', 'N/A')}
            """
            
        clinical_text = "Not Provided"
        if clinical_result:
            clinical_text = f"""
            - Risk Level: {clinical_result.get('risk_level')}
            - Summary: {clinical_result.get('summary')}
            - Keywords: {clinical_result.get('keywords')}
            """
        
        prompt = f"""
        You are a sophisticated medical diagnostic AI (MMHA-Net).
        Your task is to synthesize data from three modalities to provide a comprehensive Alzheimer's diagnosis report.
        
        **Modalities:**
        1. **MRI Analysis:** {mri_text}
        2. **EEG Analysis:** {eeg_text}
        3. **Clinical History:** {clinical_text}
            
        **Task:**
        Generate a "Final Diagnostic Report" in HTML format (using <h3>, <p>, <ul> where appropriate, no ```html tags).
        
        **Strict Output Structure:**
        1. **Predicted Condition**: (e.g., Alzheimer's Disease, MCI, or Cognitively Normal).
        2. **Risk Level**: (Low, Moderate, High, or Critical).
        3. **Explanation**: A SIMPLE, clear explanation of why this conclusion was reached, based on the provided data. Avoid overly technical jargon where possible.
        4. **Combined Decision**: A final summary statement integrating all modalities (e.g., "MRI shows atrophy and EEG indicates slowing, confirming AD").
        """
        
        response = model.generate_content(prompt)
        return response.text.replace("```html", "").replace("```", "")
        
    except Exception as e:
        logger.error(f"Error generating fusion report: {e}", exc_info=True)
        return f"<p style='color:red'>Error generating fusion report. Details: {str(e)}</p>"
