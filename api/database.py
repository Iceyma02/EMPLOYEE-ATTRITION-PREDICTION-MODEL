# api/database.py
import json
from datetime import datetime
import os

# Simple file-based logging (no database needed)
LOG_FILE = "prediction_logs.json"

def log_prediction(input_data, prediction, probability, model_version):
    """Log prediction to a JSON file"""
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": input_data,
        "prediction": prediction,
        "probability": probability,
        "model_version": model_version
    }
    
    # Read existing logs
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    
    # Add new log
    logs.append(log_entry)
    
    # Keep only last 1000 logs
    if len(logs) > 1000:
        logs = logs[-1000:]
    
    # Write back
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)
    
    return True
