import shap
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

class HeartDiseaseExplainer:
    """
    Class to calculate SHAP values for heart disease model predictions.
    Supports both individual (local) and dataset-wide (global) explanations.
    """
    def __init__(self, model, X_reference_path="models/X_reference.joblib"):
        self.model = model
        # Use TreeExplainer for XGBoost models as it's optimized for tree-based algorithms
        self.explainer = shap.TreeExplainer(self.model)
        
        # Load reference data for global explanations if available
        self.X_reference = None
        if os.path.exists(X_reference_path):
            self.X_reference = joblib.load(X_reference_path)
    
    def get_explanations(self, data: pd.DataFrame):
        """
        Computes SHAP values for the given data.
        """
        shap_values = self.explainer(data)
        return shap_values

    def get_global_explanation(self):
        """
        Generates a SHAP summary plot representing global feature importance.
        Returns a matplotlib figure.
        """
        if self.X_reference is None:
            return None
        
        shap_values_global = self.explainer(self.X_reference)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_global, self.X_reference, show=False)
        # Avoid tight_layout() as it can trigger Matplotlib mathtext errors with SHAP labels
        # plt.tight_layout() 
        
        return fig

    def get_patient_comparison(self, input_df: pd.DataFrame, healthy_baseline: dict):
        """
        Compares current patient vitals against the average healthy baseline.
        Returns a DataFrame of deltas.
        """
        patient_data = input_df.iloc[0].to_dict()
        
        comparison = []
        # Key clinical metrics to compare
        keys = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
        for key in keys:
            p_val = patient_data.get(key)
            h_val = healthy_baseline.get(key)
            delta = p_val - h_val
            
            # Simple interpretation
            status = "Above Average" if delta > 0 else "Below Average"
            if key == 'thalach': # Higher is usually better for heart rate capacity
                status = "Above Average (Optimal)" if delta > 0 else "Below Average (Higher Risk)"
            elif key == 'oldpeak' or key == 'chol' or key == 'trestbps':
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
