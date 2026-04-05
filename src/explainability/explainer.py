import shap
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

class HeartDiseaseExplainer:
    """
    Class to calculate SHAP values for heart disease model predictions.
    Supports both individual (local) and dataset-wide (global) explanations.
    """
    def __init__(self, model, preprocessor=None, X_reference_path="models/X_reference.joblib"):
        self.model = model
        self.preprocessor = preprocessor
        
        # Extract underlying tree model
        target_model = self.model
        if hasattr(self.model, 'estimator'):
            target_model = self.model.estimator
            if hasattr(target_model, 'estimator'):
                target_model = target_model.estimator
            elif hasattr(target_model, 'base_estimator'):
                target_model = target_model.base_estimator
        elif hasattr(self.model, 'base_estimator'):
            target_model = self.model.base_estimator
            
        self.explainer = shap.TreeExplainer(target_model)
        
        # Load reference data
        self.X_reference = None
        if os.path.exists(X_reference_path):
            self.X_reference = joblib.load(X_reference_path)
            
        # Initialize LIME Explainer
        self.lime_explainer = None
        if self.X_reference is not None:
            self.lime_explainer = LimeTabularExplainer(
                training_data=self.X_reference.values,
                feature_names=self.X_reference.columns.tolist(),
                class_names=['Negative', 'Positive'],
                mode='classification',
                random_state=42
            )
    
    def get_explanations(self, data: pd.DataFrame):
        """
        Computes SHAP values. Automatically pre-processes data if a preprocessor is available.
        """
        if self.preprocessor is not None:
            data = self.preprocessor.transform(data)
            
        shap_values = self.explainer(data)
        if hasattr(data, 'columns'):
            shap_values.feature_names = data.columns.tolist()
        return shap_values
        
    def get_lime_explanation(self, data_df: pd.DataFrame, num_features=10):
        """
        Generates a LIME local explanation.
        """
        if self.lime_explainer is None:
            return None
            
        # Transform data if preprocessor available
        if self.preprocessor is not None:
            data_df = self.preprocessor.transform(data_df)
            
        sample = data_df.values[0]
        
        def predict_fn(x):
            df_x = pd.DataFrame(x, columns=self.X_reference.columns)
            return self.model.predict_proba(df_x)
            
        exp = self.lime_explainer.explain_instance(
            data_row=sample,
            predict_fn=predict_fn,
            num_features=num_features
        )
        return exp

    def get_reasoning_summary(self, shap_values, feature_names=None):
        """
        Generates a reasoning summary. Use shap_values.feature_names if available.
        """
        vals = shap_values.values[0] 
        abs_vals = np.abs(vals)
        
        # Use provided names or SHAP's internal names
        names = feature_names if feature_names is not None else shap_values.feature_names
        
        sorted_indices = np.argsort(abs_vals)[::-1]
        
        top_factors = []
        for i in sorted_indices[:3]:
            feat = names[i]
            # Clean up OHE names for better readability (e.g., cat__sex_1.0 -> SEX)
            clean_feat = feat.replace("cat__", "").replace("num__", "").split("_")[0].upper()
            impact_dir = "increasing" if vals[i] > 0 else "decreasing"
            top_factors.append(f"{clean_feat} ({impact_dir})")
            
        return f"Top 3 factors contributing to risk: {', '.join(top_factors)}."

    def get_global_explanation(self):
        """
        Summary plot for global feature importance.
        """
        if self.X_reference is None:
            return None
        
        shap_values_global = self.explainer(self.X_reference)
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        shap.summary_plot(shap_values_global, self.X_reference, show=False)
        return fig

    def get_patient_comparison(self, input_df: pd.DataFrame, healthy_baseline: dict):
        """
        Compares against healthy baseline (uses original feature names).
        """
        patient_data = input_df.iloc[0].to_dict()
        comparison = []
        keys = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
        for key in keys:
            p_val = patient_data.get(key)
            h_val = healthy_baseline.get(key)
            if p_val is None or h_val is None: continue
            
            delta = p_val - h_val
            status = "Above Average" if delta > 0 else "Below Average"
            if key == 'thalach':
                status = "Above Average (Optimal)" if delta > 0 else "Below Average (Higher Risk)"
            elif key in ['oldpeak', 'chol', 'trestbps']:
                status = "Elevated Risk" if delta > 10 or (key == 'oldpeak' and delta > 0.5) else "Normal Range"
                
            comparison.append({
                "Metric": key.upper(),
                "Patient Value": p_val,
                "Healthy Median": round(h_val, 1),
                "Delta": f"{'+' if delta > 0 else ''}{round(delta, 1)}",
                "Comparison": status
            })
            
        return pd.DataFrame(comparison)

if __name__ == "__main__":
    # Test script
    MODEL_PATH = "models/heart_disease_model.joblib"
    DATA_PATH = "data/processed/heart_disease_cleaned.csv"
    
    if os.path.exists(DATA_PATH) and os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        X = df.drop("target", axis=1)
        
        explainer = HeartDiseaseExplainer(model)
        sample = X.iloc[[0]]
        shap_vals = explainer.get_explanations(sample)
        
        print(f"SHAP values for first sample: {shap_vals.values}")
        print("Base value:", shap_vals.base_values)
    else:
        print("Model or data not found. Run trainer.py first.")
