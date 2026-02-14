#!/bin/bash
# render-build.sh

echo "🚀 Installing dependencies..."
pip install -r requirements.txt

echo "📁 Checking for model files..."
if [ ! -f "models/best_model.pkl" ]; then
    echo "⚠️ Model files not found! Creating placeholder..."
    mkdir -p models
    # You need to actually have your model files in git!
    # Either commit them OR download from somewhere
fi

echo "✅ Build complete!"
