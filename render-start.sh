#!/bin/bash
# render-start.sh

echo "🚀 Starting Attrition Prediction API..."

# Check if model files exist
if [ ! -f "models/best_model.pkl" ]; then
    echo "⚠️ Model files not found! Creating placeholder..."
    mkdir -p models
    # You could download from a backup location if needed
fi

# Start the API
uvicorn api.main:app --host 0.0.0.0 --port $PORT
