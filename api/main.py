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
import traceback

# Configure logging to be VERY verbose
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
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

# Get paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models', 'model_registry', 'best_model.pkl')
preprocessor_path = os.path.join(BASE_DIR, 'models', 'model_registry', 'preprocessor.pkl')

print("\n" + "="*60)
print("🚀 ULTRA DEBUG - MODEL LOADING ATTEMPT")
print("="*60)
print(f"🔍 Python version: {sys.version}")
print(f"🔍 Current directory: {os.getcwd()}")
print(f"🔍 BASE_DIR: {BASE_DIR}")
print(f"🔍 Model path: {model_path}")
print(f"🔍 Preprocessor path: {preprocessor_path}")
print(f"🔍 Does model file exist? {os.path.exists(model_path)}")
print(f"🔍 Does preprocessor file exist? {os.path.exists(preprocessor_path)}")

if os.path.exists(model_path):
    print(f"🔍 Model file size: {os.path.getsize(model_path)} bytes")
if os.path.exists(preprocessor_path):
    print(f"🔍 Preprocessor file size: {os.path.getsize(preprocessor_path)} bytes")

# List all files in models directory
models_dir = os.path.join(BASE_DIR, 'models')
if os.path.exists(models_dir):
    print(f"\n📁 Contents of models directory:")
    for root, dirs, files in os.walk(models_dir):
        level = root.replace(models_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}📂 {os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for f in files:
            print(f"{subindent}📄 {f}")

# Print installed package versions
print("\n📦 Installed package versions:")
packages = ['numpy', 'scikit-learn', 'joblib', 'pandas', 'xgboost']
for package in packages:
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'unknown')
        print(f"   {package}: {version}")
    except ImportError as e:
        print(f"   {package}: NOT INSTALLED - {e}")

print("\n" + "="*60)

# Load model and preprocessor
model = None
preprocessor = None

try:
    print("\n🔄 Attempting to load preprocessor first...")
    preprocessor = joblib.load(preprocessor_path)
    print("✅ Preprocessor loaded successfully!")
    print(f"   Preprocessor type: {type(preprocessor)}")
    
    print("\n🔄 Attempting to load model...")
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
    print(f"   Model type: {type(model)}")
    
    # Test the model
    if hasattr(model, 'n_features_in_'):
        print(f"   Model expects {model.n_features_in_} features")
    if hasattr(preprocessor, 'feature_names_in_'):
        print(f"   Preprocessor feature names: {list(preprocessor.feature_names_in_)}")
        
except Exception as e:
    print(f"\n❌ ERROR LOADING MODEL:")
    print(f"   {type(e).__name__}: {e}")
    print("\n📋 Full traceback:")
    traceback.print_exc()
    print("\n" + "="*60)

print("\n" + "="*60)
print(f"🏁 Final status - Model loaded: {model is not None}")
print(f"🏁 Final status - Preprocessor loaded: {preprocessor is not None}")
print("="*60 + "\n")

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
    """Debug endpoint to check file structure"""
    import glob
    debug_info = {
        "current_dir": os.getcwd(),
        "base_dir": BASE_DIR,
        "model_path": str(model_path),
        "model_exists": os.path.exists(model_path),
        "preprocessor_path": str(preprocessor_path),
        "preprocessor_exists": os.path.exists(preprocessor_path),
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
    }
    
    # Get all .pkl files recursively
    pkl_files = glob.glob(os.path.join(BASE_DIR, '**', '*.pkl'), recursive=True)
    debug_info["all_pkl_files"] = pkl_files
    
    return debug_info

@app.post("/predict", response_model=PredictionResponse)
async def predict_attrition(employee: EmployeeData):
    """Predict employee attrition probability"""
    
    if model is None or preprocessor is None:
        logger.error("Prediction attempted but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_dict = employee.dict()
        logger.debug(f"Received prediction request")
        
        input_df = pd.DataFrame([input_dict])
        logger.debug(f"Input shape: {input_df.shape}")
        
        processed = preprocessor.transform(input_df)
        logger.debug(f"Processed shape: {processed.shape}")
        
        prediction = model.predict(processed)[0]
        logger.debug(f"Prediction: {prediction}")
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(processed)[0]
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            probability = float(prediction)
        
        risk_level = "High" if probability > 0.6 else "Medium" if probability > 0.3 else "Low"
        confidence_score = probability if prediction == 1 else 1 - probability
        
        logger.info(f"Prediction successful: {risk_level} risk ({probability:.2f})")
        
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
    return {
        "message": "Employee Attrition Prediction API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "/health": "GET - Check API health",
            "/predict": "POST - Predict attrition risk",
            "/debug/files": "GET - Debug file structure",
            "/docs": "GET - Interactive API documentation"
        }
    }
