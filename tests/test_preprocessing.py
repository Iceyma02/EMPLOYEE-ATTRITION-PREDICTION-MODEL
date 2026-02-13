import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.preprocessing import load_raw_data, create_preprocessing_pipeline, prepare_data

def test_load_raw_data():
    """Test that data loads correctly"""
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
        df, preprocessor, drop_cols
    )
    assert X_train.shape[0] > X_test.shape[0]
    assert len(feature_names) > 0
