# tests/test_preprocessing.py
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessing import load_raw_data, create_preprocessing_pipeline, prepare_data

def test_load_raw_data():
    """Test that data loads correctly"""
    # Use a smaller test path or mock
    df = load_raw_data()
    assert df.shape[0] > 0
    assert 'Attrition' in df.columns

def test_preprocessing_pipeline():
    """Test pipeline creation"""
    preprocessor, drop_cols, num_cols, cat_cols = create_preprocessing_pipeline()
    assert preprocessor is not None
    assert len(num_cols) > 0
    assert len(cat_cols) > 0

def test_data_preparation():
    """Test train/test split"""
    df = load_raw_data()
    preprocessor, drop_cols, _, _ = create_preprocessing_pipeline()
    X_train, X_test, y_train, y_test, feature_names, fitted = prepare_data(
        df, preprocessor, drop_cols, test_size=0.2
    )
    assert X_train.shape[0] > X_test.shape[0]
    assert len(feature_names) > 0
