# api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import schemas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import EmployeeData, PredictionResponse
from api.database import log_prediction

app = FastAPI(title="Attrition Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the correct paths - using model_registry folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models', 'model_registry', 'best_model.pkl')
preprocessor_path = os.path.join(BASE_DIR, 'models', 'model_registry', 'preprocessor.pkl')

print(f"🔍 Looking for model at: {model_path}")
print(f"🔍 Looking for preprocessor at: {preprocessor_path}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"BASE_DIR: {BASE_DIR}")

# Check if models directory exists
models_dir = os.path.join(BASE_DIR, 'models')
if os.path.exists(models_dir):
    logger.info(f"✅ models directory exists at: {models_dir}")
    logger.info(f"Contents of models: {os.listdir(models_dir)}")
else:
    logger.error(f"❌ models directory NOT found at: {models_dir}")

# Check if model_registry exists
model_registry_dir = os.path.join(models_dir, 'model_registry')
if os.path.exists(model_registry_dir):
    logger.info(f"✅ model_registry directory exists at: {model_registry_dir}")
    logger.info(f"Contents of model_registry: {os.listdir(model_registry_dir)}")
else:
    logger.error(f"❌ model_registry directory NOT found at: {model_registry_dir}")

# Load model and preprocessor
try:
    # Check if files exist before loading
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file does NOT exist at: {model_path}")
    else:
        logger.info(f"✅ Model file exists at: {model_path}")
        file_size = os.path.getsize(model_path)
        logger.info(f"Model file size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
    
    if not os.path.exists(preprocessor_path):
        logger.error(f"❌ Preprocessor file does NOT exist at: {preprocessor_path}")
    else:
        logger.info(f"✅ Preprocessor file exists at: {preprocessor_path}")
        file_size = os.path.getsize(preprocessor_path)
        logger.info(f"Preprocessor file size: {file_size} bytes ({file_size/1024:.2f} KB)")
    
    # Only attempt to load if both files exist
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"✅ Model loaded successfully from: {model_path}")
        logger.info(f"✅ Preprocessor loaded successfully from: {preprocessor_path}")
        
        # Log model information
        logger.info(f"Model type: {type(model).__name__}")
        if hasattr(model, 'n_features_in_'):
            logger.info(f"Model expects {model.n_features_in_} features")
        if hasattr(model, 'classes_'):
            logger.info(f"Model classes: {model.classes_}")
    else:
        logger.error("❌ Cannot load model - files missing")
        model = None
        preprocessor = None
    
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    logger.exception("Detailed traceback:")
    model = None
    preprocessor = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        return {
            "status": "degraded",
            "model": "not loaded",
            "error": "Model failed to load",
            "model_path": str(model_path),
            "model_exists": os.path.exists(model_path),
            "preprocessor_exists": os.path.exists(preprocessor_path)
        }
    
    return {
        "status": "healthy",
        "model": "Attrition_Predictor",
        "model_path": str(model_path),
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/debug/files")
async def debug_files():
    """Temporary debug endpoint to check file structure"""
    debug_info = {
        "current_dir": os.getcwd(),
        "base_dir": BASE_DIR,
        "models_dir": models_dir,
        "models_dir_exists": os.path.exists(models_dir),
        "model_registry_dir": model_registry_dir,
        "model_registry_exists": os.path.exists(model_registry_dir),
        "model_file": str(model_path),
        "model_file_exists": os.path.exists(model_path),
        "preprocessor_file": str(preprocessor_path),
        "preprocessor_file_exists": os.path.exists(preprocessor_path),
    }
    
    # Try to list directories if they exist
    if debug_info["models_dir_exists"]:
        try:
            debug_info["models_contents"] = os.listdir(models_dir)
        except Exception as e:
            debug_info["models_contents"] = f"Error listing: {str(e)}"
    
    if debug_info["model_registry_exists"]:
        try:
            debug_info["registry_contents"] = os.listdir(model_registry_dir)
        except Exception as e:
            debug_info["registry_contents"] = f"Error listing: {str(e)}"
    
    # Add file sizes if they exist
    if debug_info["model_file_exists"]:
        try:
            debug_info["model_file_size"] = os.path.getsize(model_path)
        except:
            debug_info["model_file_size"] = "Error getting size"
    
    if debug_info["preprocessor_file_exists"]:
        try:
            debug_info["preprocessor_file_size"] = os.path.getsize(preprocessor_path)
        except:
            debug_info["preprocessor_file_size"] = "Error getting size"
    
    return debug_info

@app.post("/predict", response_model=PredictionResponse)
async def predict_attrition(employee: EmployeeData):
    """Predict employee attrition probability"""
    
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = employee.dict()
        logger.info(f"Received prediction request for employee: Age={input_dict.get('Age')}, Role={input_dict.get('JobRole')}")
        
        input_df = pd.DataFrame([input_dict])
        
        # Preprocess
        processed = preprocessor.transform(input_df)
        logger.info(f"Preprocessed data shape: {processed.shape}")
        
        # Predict
        prediction = model.predict(processed)[0]
        logger.info(f"Raw prediction: {prediction}")
        
        # Get probability if model supports it
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed)[0]
            probability = probabilities[1]  # Probability of attrition (class 1)
            logger.info(f"Prediction probabilities: Stay={probabilities[0]:.3f}, Leave={probabilities[1]:.3f}")
        else:
            probability = float(prediction)  # fallback
            logger.warning("Model does not support predict_proba, using fallback")
        
        # Log to database (optional)
        try:
            log_prediction(
                input_data=input_dict,
                prediction=int(prediction),
                probability=float(probability),
                model_version="v1.0"
            )
        except Exception as log_error:
            logger.warning(f"Logging failed (non-critical): {log_error}")
        
        # Determine risk level
        if probability > 0.6:
            risk_level = "High"
        elif probability > 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Calculate confidence score
        if prediction == 1:
            confidence_score = float(probability)
        else:
            confidence_score = float(1 - probability)
        
        logger.info(f"Prediction result: {risk_level} risk with {probability:.2f} probability")
        
        return PredictionResponse(
            attrition_risk=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            confidence_score=confidence_score,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.exception("Detailed traceback:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Employee Attrition Prediction API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "/health": "GET - Check API health",
            "/predict": "POST - Predict attrition risk",
            "/debug/files": "GET - Debug file structure (temporary)",
            "/docs": "GET - Interactive API documentation"
        }
    }
