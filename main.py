import os
import pandas as pd
from src.data.loader import load_raw_data, clean_data, save_processed_data
from src.models.trainer import train_model, save_model_artifacts

def run_pipeline(tune=True):
    """
    Orchestrates the entire ML pipeline from data loading to model training.
    """
    print("--- Heart Disease Risk Prediction Pipeline ---")
    
    # 1. Data Paths
    RAW_DATA_PATH = "data/raw/heart_disease_cleveland.csv"
    PROCESSED_DATA_PATH = "data/processed/heart_disease_cleaned.csv"
    MODEL_DIR = "models"
    
    # 2. Load and Clean Data
    print("\nPhase 1: Data Integration")
    raw_df = load_raw_data(RAW_DATA_PATH)
    print(f"Loaded {len(raw_df)} rows from {RAW_DATA_PATH}")
    
    cleaned_df = clean_data(raw_df)
    save_processed_data(cleaned_df, PROCESSED_DATA_PATH)
    
    # 3. Model Training & Optimization
    print("\nPhase 2: Model Training & Optimization")
    X = cleaned_df.drop("target", axis=1)
    y = cleaned_df["target"]
    
    model, metrics = train_model(X, y, tune=tune)
    save_model_artifacts(model, metrics, MODEL_DIR)
    
    print("\nPipeline execution complete.")

if __name__ == "__main__":
    # Set tune=True for hyperparameter optimization
    run_pipeline(tune=True)
