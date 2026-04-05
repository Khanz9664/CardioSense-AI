import pandas as pd
import numpy as np

class HeartDiseaseSafetyEngine:
    """
    Advanced Medical Safety Engine for Heart Disease CDSS.
    Integrates clinical guidelines, prediction overrides, and robust uncertainty estimation.
    """
    def __init__(self):
        # Training distribution bounds (from model_metadata.json analysis)
        self.training_bounds = {
            'age': (29.0, 77.0),
            'trestbps': (94.0, 200.0),
            'chol': (126.0, 564.0),
            'thalach': (71.0, 202.0),
            'oldpeak': (0.0, 6.2)
        }

    def check_out_of_distribution(self, input_df: pd.DataFrame):
        """
        Checks if current inputs fall outside the known training distribution.
        """
        warnings = []
        data = input_df.iloc[0].to_dict()
        
        for feature, (min_val, max_val) in self.training_bounds.items():
            current_val = data.get(feature)
            if current_val is not None:
                if current_val < min_val:
                    warnings.append(f"{feature.upper()} ({current_val}) is BELOW the training minimum ({min_val}).")
                elif current_val > max_val:
                    warnings.append(f"{feature.upper()} ({current_val}) is ABOVE the training maximum ({max_val}).")
        
        return warnings

    def calculate_confidence(self, probability: float):
        """
        Calculates confidence based on binary entropy.
        Normalized confidence = 1 - H(p).
        """
        p = np.clip(probability, 1e-7, 1 - 1e-7)
        entropy = - (p * np.log2(p) + (1 - p) * np.log2(1 - p))
        
        # Normalized confidence (0 to 1)
        confidence_score = 1 - entropy
        
        if confidence_score > 0.8:
            level = "HIGH"
            color = "#28a745"
            description = "The model's probability distribution is sharply focused. Statistical certainty is robust."
        elif confidence_score > 0.4:
            level = "MODERATE"
            color = "#fcc419"
            description = "The prediction carries aleatoric uncertainty. Results are mathematically less distinct."
        else:
            level = "LOW"
            color = "#ff4b4b"
            description = "High entropy (ambiguity). The AI cannot distinguish risk levels with high certainty."
            
        return {
            "level": level,
            "color": color,
            "description": description,
            "score": round(confidence_score * 100, 1),
            "entropy": round(entropy, 3)
        }

    def get_clinical_assessment(self, input_df: pd.DataFrame):
        """
        Analyzes vitals against AHA/ACC standards.
        """
        results = []
        data = input_df.iloc[0].to_dict()
        
        # Hypertension Analysis (AHA Guidelines)
        bp = data.get('trestbps', 120)
        if bp >= 180:
            results.append({"factor": "Blood Pressure", "status": "HYPERTENSIVE CRISIS", "severity": "CRITICAL"})
        elif bp >= 140:
            results.append({"factor": "Blood Pressure", "status": "Stage 2 Hypertension", "severity": "HIGH"})
        elif bp >= 130:
            results.append({"factor": "Blood Pressure", "status": "Stage 1 Hypertension", "severity": "MODERATE"})
        
        # Cholesterol Analysis
        chol = data.get('chol', 200)
        if chol >= 240:
            results.append({"factor": "Cholesterol", "status": "Hypercholesterolemia (>240 mg/dL)", "severity": "HIGH"})
        elif chol >= 200:
            results.append({"factor": "Cholesterol", "status": "Borderline High", "severity": "MODERATE"})
        
        # Fasting Blood Sugar
        if data.get('fbs') == 1:
            results.append({"factor": "Metabolic", "status": "Hyperglycemia (FBS > 120)", "severity": "MODERATE"})
            
        return results

    def get_clinical_overrides(self, input_df: pd.DataFrame):
        """
        Clinical Risk Override System.
        Escalates prediction if critical life-safety thresholds are breached.
        """
        overrides = []
        data = input_df.iloc[0].to_dict()
        
        if data.get('trestbps', 120) >= 180:
            overrides.append({
                "reason": "Systolic BP >= 180 (Hypertensive Crisis). Immediate risk escalation.",
                "forced_risk": 1
            })
            
        if data.get('ca', 0) >= 2:
            overrides.append({
                "reason": "Multivessel Disease (CA >= 2 major vessels). Structural risk override.",
                "forced_risk": 1
            })
            
        if data.get('oldpeak', 0) > 3.0:
            overrides.append({
                "reason": "Severe Ischemic Depression (Oldpeak > 3.0). Clinical severity override.",
                "forced_risk": 1
            })
            
        return overrides

if __name__ == "__main__":
    safety = HeartDiseaseSafetyEngine()
    test_df = pd.DataFrame({'age': [15.0], 'chol': [800.0]})
    print(f"OOD Warnings: {safety.check_out_of_distribution(test_df)}")
    print(f"Confidence (0.55): {safety.calculate_confidence(0.55)}")
    print(f"Confidence (0.95): {safety.calculate_confidence(0.95)}")
