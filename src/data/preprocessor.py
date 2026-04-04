import pandas as pd
import joblib
import os

class HeartDiseasePreprocessor:
    """
    Unified preprocessing class to ensure consistency between training 
    and real-time inference (API/UI).
    """
    def __init__(self):
        # Strictly define columns to ensure model input consistency
        self.feature_order = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 
            'slope', 'ca', 'thal'
        ]
        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        """
        Learns parameters from the training data (e.g., column names).
        In this case, it primarily validates the schema.
        """
        # Ensure all required features are present
        missing = set(self.feature_order) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame):
        """
        Applies transformations and enforces feature ordering.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet.")
        
        # Select and reorder columns
        X_copy = X[self.feature_order].copy()
        
        # Ensure correct numeric types
        for col in self.feature_order:
            X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')
        
        return X_copy

    def save(self, path: str):
        """
        Persists the preprocessor object to a joblib file.
        """
        joblib.dump(self, path)
        print(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: str):
        """
        Loads a persisted preprocessor.
        """
        if os.path.exists(path):
            return joblib.load(path)
        else:
            raise FileNotFoundError(f"Preprocessor file not found at {path}")

if __name__ == "__main__":
    # Quick Test
    data = pd.DataFrame({
        'age': [50], 'sex': [1], 'cp': [2], 'trestbps': [120], 'chol': [200],
        'fbs': [0], 'restecg': [1], 'thalach': [150], 'exang': [0],
        'oldpeak': [1.0], 'slope': [1], 'ca': [0], 'thal': [3]
    })
    
    preprocessor = HeartDiseasePreprocessor()
    preprocessor.fit(data)
    transformed = preprocessor.transform(data)
    print("Transformed data shape:", transformed.shape)
    print("Feature Order:", transformed.columns.tolist())
