import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class HeartDiseasePreprocessor:
    """
    Production-grade preprocessing pipeline using ColumnTransformer and Pipeline.
    Ensures numerical scaling and categorical encoding consistency.
    """
    def __init__(self):
        self.feature_order = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 
            'slope', 'ca', 'thal'
        ]
        self.num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        self.cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        
        self.pipeline = self._build_pipeline()
        self.is_fitted = False
        self.feature_names_out = None

    def _build_pipeline(self):
        """
        Constructs the internal Scikit-Learn pipeline.
        """
        num_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.num_features),
                ('cat', cat_transformer, self.cat_features)
            ],
            remainder='passthrough'
        )

        return Pipeline(steps=[('preprocessor', preprocessor)])

    def fit(self, X: pd.DataFrame):
        """
        Fits the robust pipeline to the training data.
        """
        self.pipeline.fit(X[self.feature_order])
        self.is_fitted = True
        
        # Capture the generated feature names after OneHotEncoding
        ct = self.pipeline.named_steps['preprocessor']
        self.feature_names_out = ct.get_feature_names_out()
        return self

    def transform(self, X: pd.DataFrame):
        """
        Applies the fitted pipeline transformations.
        Returns a DataFrame with updated feature names.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet.")
        
        X_copy = X[self.feature_order].copy()
        transformed_data = self.pipeline.transform(X_copy)
        
        return pd.DataFrame(transformed_data, columns=self.feature_names_out, index=X.index)

    def save(self, path: str):
        """
        Persists the entire preprocessor object.
        """
        joblib.dump(self, path)
        print(f"Robust Preprocessor saved to {path}")

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
