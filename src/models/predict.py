import joblib
import pandas as pd
import os

from src.data.preprocessor import HeartDiseasePreprocessor

class HeartDiseasePredictor:
    """
    Class to load the trained model and perform predictions.
    Integrates the unified preprocessor for consistent feature engineering.
    """
    def __init__(self, model_path: str, preprocessor_path: str = "models/preprocessor.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)
        
        # Load the unified preprocessor
        if os.path.exists(preprocessor_path):
            self.preprocessor = HeartDiseasePreprocessor.load(preprocessor_path)
        else:
            # Fallback (non-fitted) preprocessor if file is missing during init
            self.preprocessor = HeartDiseasePreprocessor()
    
    def predict(self, input_data: pd.DataFrame):
        """
        Expects a DataFrame. Processes it through the unified preprocessor 
        before performing inference.
        Returns prediction labels and probabilities.
        """
        # 1. Apply unified preprocessing (reordering, typing, etc.)
        processed_data = self.preprocessor.transform(input_data)
        
        # 2. Model Inference
        prediction = self.model.predict(processed_data)
        probability = self.model.predict_proba(processed_data)
        
        return prediction, probability

if __name__ == "__main__":
    # Test with sample from processed data
    MODEL_PATH = "models/heart_disease_model.joblib"
    DATA_PATH = "data/processed/heart_disease_cleaned.csv"
    
    if os.path.exists(DATA_PATH) and os.path.exists(MODEL_PATH):
        df = pd.read_csv(DATA_PATH)
        sample = df.drop("target", axis=1).head(5)
        
        predictor = HeartDiseasePredictor(MODEL_PATH)
        preds, probs = predictor.predict(sample)
        print("Predictions:", preds)
        print("Probabilities:", probs)
    else:
        print("Model or data not found. Run trainer.py first.")
