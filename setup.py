# setup.py
from setuptools import setup, find_packages

setup(
    name="employee-attrition-prediction",
    version="1.0.0",
    description="Employee Attrition Prediction System for CBZ Holdings",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "streamlit>=1.25.0",
        "joblib>=1.3.0",
    ],
    python_requires=">=3.9",
)
