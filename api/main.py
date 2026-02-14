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

# Get the correct paths - UPDATED to use model_registry folder
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

# Rest of your code remains exactly the same...
# [Keep all your existing endpoints exactly as they are]
