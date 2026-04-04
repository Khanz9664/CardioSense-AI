import pandas as pd
import numpy as np
import copy

class HeartDiseaseSimulator:
    """
    Class to simulate real-time patient data changes and assess impact on risk.
    Supports multi-variable baseline comparison and automated recommendations.
    """
    def __init__(self, model):
        self.model = model
        # Clinical targets for modifiable risk factors
        self.targets = {
            'chol': 200,      # mg/dl
            'trestbps': 120,  # mm Hg
            'thalach': 165,   # Higher is generally better for thalach
            'oldpeak': 0.0    # No ST depression
        }
    
    def simulate_multi_change(self, base_data: pd.DataFrame, updates: dict):
        """
        Takes a base data point, modifies multiple features, and calculates new risk.
        Updates should be a dictionary like {'chol': 200, 'trestbps': 120}
        """
        sim_data = base_data.copy()
        original_prob = self.model.predict_proba(base_data)[:, 1][0]
        
        # Apply all updates
        for feature, new_value in updates.items():
            if feature in sim_data.columns:
                sim_data[feature] = new_value
        
        new_prob = self.model.predict_proba(sim_data)[:, 1][0]
        delta = new_prob - original_prob
        
        return {
            "original_prob": original_prob,
            "new_prob": new_prob,
            "delta": delta,
            "updates": updates
        }

    def generate_recommendations(self, base_data: pd.DataFrame, shap_values):
        """
        Identifies key risk drivers using SHAP and calculates the potential risk reduction
        if modifiable factors were moved to clinical targets.
        """
        # 1. Get modifiable feature names
        modifiable = ['chol', 'trestbps', 'thalach', 'oldpeak']
        
        # 2. Extract feature importance/impact from SHAP for this specific patient
        # SHAP values indices match column names
        feature_names = base_data.columns.tolist()
        patient_shap = shap_values.values[0]
        
        recommendations = []
        original_prob = self.model.predict_proba(base_data)[:, 1][0]

        for feature in modifiable:
            idx = feature_names.index(feature)
            impact = patient_shap[idx]
            current_val = base_data[feature].iloc[0]
            target_val = self.targets[feature]

            # Only suggest if it actually contributes to risk OR is sub-optimal
            is_suboptimal = False
            if feature == 'thalach':
                is_suboptimal = current_val < target_val
            else:
                is_suboptimal = current_val > target_val

            if impact > 0 or is_suboptimal:
                # Simulate if we hit the target
                sim_result = self.simulate_multi_change(base_data, {feature: target_val})
                new_prob = sim_result['new_prob']
                
                if new_prob < original_prob:
                    recommendations.append({
                        "feature": feature,
                        "current": current_val,
                        "target": target_val,
                        "original_risk": original_prob,
                        "new_risk": new_prob,
                        "impact_rank": impact
                    })

        # Sort recommendations by highest potential impact (risk reduction)
        recommendations = sorted(recommendations, key=lambda x: x['new_risk'])
        
        return recommendations

if __name__ == "__main__":
    import joblib
    import os
    
    MODEL_PATH = "models/heart_disease_model.joblib"
    DATA_PATH = "data/processed/heart_disease_cleaned.csv"
    
    if os.path.exists(DATA_PATH) and os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        X = df.drop("target", axis=1)
        base_patient = X.iloc[[0]]
        
        simulator = HeartDiseaseSimulator(model)
        
        # Test: Reduce Cholesterol
        print("Test: Reducing Cholesterol from", base_patient['chol'].iloc[0], "to 200")
        result = simulator.simulate_multi_change(base_patient, {'chol': 200})
        print(f"Risk Probability Change: {result['delta']:.4f}")
    else:
        print("Model or data not found. Run trainer.py first.")
