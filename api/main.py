# api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
import sys

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

# Load model and preprocessor
try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print(f"✅ Model loaded from: {model_path}")
    print(f"✅ Preprocessor loaded from: {preprocessor_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    preprocessor = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        return {
            "status": "degraded",
            "model": "not loaded",
            "error": "Model failed to load"
        }
    
    return {
        "status": "healthy",
        "model": "Attrition_Predictor",
        "model_path": str(model_path),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_attrition(employee: EmployeeData):
    """Predict employee attrition probability"""
    
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = employee.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Preprocess
        processed = preprocessor.transform(input_df)
        
        # Predict
        prediction = model.predict(processed)[0]
        
        # Get probability if model supports it
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(processed)[0][1]
        else:
            probability = float(prediction)  # fallback
        
        # Log to database (optional)
        try:
            log_prediction(
                input_data=input_dict,
                prediction=int(prediction),
                probability=float(probability),
                model_version="v1.0"
            )
        except:
            pass  # Logging failure shouldn't break prediction
        
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
        
        return PredictionResponse(
            attrition_risk=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            confidence_score=confidence_score,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Employee Attrition Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Check API health",
            "/predict": "POST - Predict attrition risk",
            "/docs": "GET - Interactive API documentation"
        }
    }
