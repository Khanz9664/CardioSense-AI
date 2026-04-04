import pandas as pd
import numpy as np
import os

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw heart disease data from a CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the heart disease dataset by imputing missing values.
    Uses median for numeric/categorical columns as a baseline strategy.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_cleaned = df.copy()
    
    # Impute missing values with median (as done in the EDA notebook)
    # Note: For 'ca' and 'thal', median is a reasonable mode-like estimate for these discrete scales.
    df_cleaned = df_cleaned.fillna(df_cleaned.median())
    
    return df_cleaned

def save_processed_data(df: pd.DataFrame, output_path: str):
    """
    Saves the cleaned dataframe to the specified path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    # Test script
    FILE_PATH = "data/raw/heart_disease_cleveland.csv"
    OUTPUT_PATH = "data/processed/heart_disease_cleaned.csv"
    
    data = load_raw_data(FILE_PATH)
    cleaned_data = clean_data(data)
    save_processed_data(cleaned_data, OUTPUT_PATH)
