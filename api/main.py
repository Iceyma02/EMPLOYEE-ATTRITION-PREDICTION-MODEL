# api/main.py - Add this COMPLETE file

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
import sys
import logging
import traceback

# Add at the VERY TOP of api/main.py, right after imports
import sys
print("🚀 STARTUP DEBUG - Python version:", sys.version)
print("🚀 STARTUP DEBUG - Current directory:", os.getcwd())
print("🚀 STARTUP DEBUG - Files in current dir:", os.listdir('.'))

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

# Global variables for model and preprocessor
model = None
preprocessor = None

# Load model and preprocessor with better error handling
try:
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        logger.info("Loading model and preprocessor...")
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"✅ Model type: {type(model).__name__}")
        logger.info(f"✅ Preprocessor type: {type(preprocessor).__name__}")
        
        # Test the model with dummy data to ensure it works
        try:
            # Create minimal dummy data for testing
            dummy_data = pd.DataFrame({col: [0] for col in preprocessor.feature_names_in_})
            test_transform = preprocessor.transform(dummy_data)
            test_pred = model.predict(test_transform)
            logger.info(f"✅ Model test successful - test prediction: {test_pred[0]}")
        except Exception as test_e:
            logger.error(f"❌ Model test failed: {test_e}")
            logger.exception("Test failure traceback:")
            model = None
            preprocessor = None
    else:
        logger.error("❌ Model files missing")
        if not os.path.exists(model_path):
            logger.error(f"Missing: {model_path}")
        if not os.path.exists(preprocessor_path):
            logger.error(f"Missing: {preprocessor_path}")
            
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    logger.exception("Detailed traceback:")
    model = None
    preprocessor = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_path": str(model_path),
        "model_exists": os.path.exists(model_path),
        "preprocessor_exists": os.path.exists(preprocessor_path),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/debug/files")
async def debug_files():
    """Temporary debug endpoint to check file structure"""
    debug_info = {
        "current_dir": os.getcwd(),
        "base_dir": BASE_DIR,
        "models_dir_exists": os.path.exists(os.path.join(BASE_DIR, 'models')),
        "model_registry_exists": os.path.exists(os.path.join(BASE_DIR, 'models', 'model_registry')),
        "model_file_exists": os.path.exists(model_path),
        "preprocessor_file_exists": os.path.exists(preprocessor_path),
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
    }
    
    # Add file sizes
    if debug_info["model_file_exists"]:
        debug_info["model_file_size"] = os.path.getsize(model_path)
    if debug_info["preprocessor_file_exists"]:
        debug_info["preprocessor_file_size"] = os.path.getsize(preprocessor_path)
    
    return debug_info

@app.post("/predict", response_model=PredictionResponse)
async def predict_attrition(employee: EmployeeData):
    """Predict employee attrition probability with crash protection"""
    
    # Check if model is loaded
    if model is None or preprocessor is None:
        logger.error("Prediction attempted but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Log incoming request (without sensitive data)
        logger.info(f"Received prediction request")
        
        # Convert input to DataFrame
        input_dict = employee.dict()
        logger.debug(f"Input data keys: {list(input_dict.keys())}")
        
        input_df = pd.DataFrame([input_dict])
        logger.debug(f"DataFrame shape: {input_df.shape}")
        
        # Preprocess with error catching
        try:
            processed = preprocessor.transform(input_df)
            logger.debug(f"Processed shape: {processed.shape}")
        except Exception as preprocess_e:
            logger.error(f"Preprocessing failed: {preprocess_e}")
            logger.exception("Preprocessing traceback:")
            raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(preprocess_e)}")
        
        # Predict with error catching
        try:
            prediction = model.predict(processed)[0]
            logger.debug(f"Raw prediction: {prediction}")
        except Exception as predict_e:
            logger.error(f"Prediction failed: {predict_e}")
            logger.exception("Prediction traceback:")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(predict_e)}")
        
        # Get probability
        probability = 0.5  # default
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(processed)[0]
                probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
                logger.debug(f"Probability: {probability}")
        except Exception as proba_e:
            logger.warning(f"Probability calculation failed: {proba_e}")
            probability = float(prediction)
        
        # Determine risk level
        if probability > 0.6:
            risk_level = "High"
        elif probability > 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Calculate confidence score
        confidence_score = probability if prediction == 1 else 1 - probability
        
        # Log prediction (optional - wrapped in try/except)
        try:
            log_prediction(
                input_data=input_dict,
                prediction=int(prediction),
                probability=float(probability),
                model_version="v1.0"
            )
        except Exception as log_e:
            logger.warning(f"Logging failed: {log_e}")
        
        logger.info(f"Prediction successful: {risk_level} risk ({probability:.2f})")
        
        return PredictionResponse(
            attrition_risk=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            confidence_score=confidence_score,
            timestamp=datetime.now()
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Employee Attrition Prediction API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "status": "running",
        "endpoints": {
            "/health": "GET - Check API health",
            "/predict": "POST - Predict attrition risk",
            "/debug/files": "GET - Debug file structure",
            "/docs": "GET - Interactive API documentation"
        }
    }
