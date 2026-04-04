import pandas as pd
import numpy as np

class HeartDiseaseSafetyEngine:
    """
    Utility class for data integrity checks and model confidence scoring.
    Ensures that the AI is acting within clinical training bounds.
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
        Returns a list of warnings or an empty list if safety checks pass.
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
        Maps the raw model probability to a qualitative confidence level.
        Confidence is highest when far from the 0.5 decision boundary.
        """
        # Distance from the threshold (0 to 0.5)
        certainty = abs(probability - 0.5) * 2 # Normalized to 0 to 1
        
        if certainty > 0.7:
            level = "HIGH"
            color = "#28a745" # Medical green
            description = "The model is very certain about this assessment."
        elif certainty > 0.3:
            level = "MODERATE"
            color = "#fcc419" # Clinical yellow
            description = "The assessment is likely correct but requires clinical review."
        else:
            level = "LOW"
            color = "#ff4b4b" # Warning red
            description = "Clinical ambiguity detected. The AI suggests manual scrutiny."
            
        return {
            "level": level,
            "color": color,
            "description": description,
            "score": round(certainty * 100, 1) # Probability mass certainty %
        }

    def get_clinical_guardrails(self, input_df: pd.DataFrame):
        """
        Detects critical clinical states that should overrule model predictions.
        """
        hard_stops = []
        data = input_df.iloc[0].to_dict()
        
        # Hypothetical clinical guardrails
        if data.get('thalach', 100) < 50:
            hard_stops.append("Critical Bradycardia (Max HR < 50). Urgent manual triage required.")
        if data.get('trestbps', 120) > 200:
            hard_stops.append("Hypertensive Crisis (BP > 200). Predictive model results may be secondary to acute risk.")
            
        return hard_stops

if __name__ == "__main__":
    safety = HeartDiseaseSafetyEngine()
    test_df = pd.DataFrame({'age': [15.0], 'chol': [800.0]})
    print(f"OOD Warnings: {safety.check_out_of_distribution(test_df)}")
    print(f"Confidence (0.55): {safety.calculate_confidence(0.55)}")
    print(f"Confidence (0.95): {safety.calculate_confidence(0.95)}")
