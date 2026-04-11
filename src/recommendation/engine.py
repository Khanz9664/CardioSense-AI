import pandas as pd
import numpy as np

class HeartDiseaseRecommender:
    """
    Advanced recommendation engine that identifies context-aware medical patterns
    and prioritizes clinical interventions based on severity.
    """
    def __init__(self):
        # Clinical thresholds for individual markers (baseline)
        self.thresholds = {
            'trestbps': 140, 
            'chol': 240,
            'oldpeak': 1.5,
            'thalach': 120
        }

    def infer_medical_patterns(self, data: pd.DataFrame):
        """
        Infers complex medical patterns by analyzing feature interactions.
        Returns a list of identified patterns.
        """
        row = data.iloc[0]
        patterns = []

        # 1. Cardiac Stress during Exercise
        if row['oldpeak'] > 2.0 and row['thalach'] < 140:
            patterns.append({
                "id": "cardiac_stress",
                "label": "Elevated Cardiac Stress Pattern",
                "severity": "High",
                "rationale": "The combination of significant ST depression (>2.0) and a relatively low maximum heart rate suggests myocardial stress during physical exertion."
            })

        # 2. Silent Ischemia Risk
        if row['exang'] == 1 and row['cp'] == 4:
            patterns.append({
                "id": "silent_ischemia",
                "label": "Potential Silent Ischemia Risk",
                "severity": "High",
                "rationale": "Exercise-induced angina in the absence of typical chest pain symptoms (Asymptomatic) is a strong indicator of silent ischemic heart disease."
            })

        # 3. Hypertensive Management
        if row['trestbps'] > 160:
            severity = "High" if row['trestbps'] > 180 else "Moderate"
            patterns.append({
                "id": "hypertension",
                "label": f"{severity} Hypertensive Markers",
                "severity": severity,
                "rationale": f"Resting blood pressure of {row['trestbps']} mm Hg is significantly above the clinical target (120/80)."
            })

        # 4. Metabolic/Lipid Profile
        if row['chol'] > 260:
            patterns.append({
                "id": "dyslipidemia",
                "label": "Severe Dyslipidemia",
                "severity": "Moderate",
                "rationale": "Serum cholesterol levels above 260 mg/dl contribute significantly to atherosclerotic plaque build-up."
            })
            
        # 5. Vascular Indicators
        if row['ca'] >= 2:
            patterns.append({
                "id": "vascular_complexity",
                "label": "High Vascular Complexity",
                "severity": "Moderate",
                "rationale": f"Fluoroscopy showing {row['ca']} major vessels involved indicates multi-vessel disease progression."
            })

        return patterns

    def generate_prioritized_recommendations(self, data: pd.DataFrame, probability: float, shap_values):
        """
        Generates prioritized recommendations by combining patterns, SHAP impact, and risk probability.
        """
        patterns = self.infer_medical_patterns(data)
        recommendations = []

        # Convert patterns to recommendation format
        for p in patterns:
            recommendations.append({
                "title": p['label'],
                "priority": p['severity'],
                "rationale": p['rationale'],
                "type": "Pattern-Based"
            })

        # Augment with SHAP-based modifiable factors if not already covered by patterns
        feature_names = shap_values.feature_names
        patient_shap = shap_values.values[0]
        
        # modifiable factors to check if not covered
        modifiable = {'chol': 'Lipid Management', 'trestbps': 'Blood Pressure Control', 'oldpeak': 'ST-Segment Analysis'}
        
        # Check SHAP for high impact features
        for feat, label in modifiable.items():
            # Find the index in transformed features (handling num__ prefix)
            try:
                idx = -1
                for i, name in enumerate(feature_names):
                    if name == feat or name == f"num__{feat}":
                        idx = i
                        break
                
                if idx != -1 and patient_shap[idx] > 0.5: # Significant risk driver
                    # Check if already covered by a pattern
                    if not any(feat in p['id'] for p in patterns):
                        recommendations.append({
                            "title": f"Optimize {label}",
                            "priority": "Moderate" if probability < 0.7 else "High",
                            "rationale": f"SHAP analysis identifies {feat.upper()} as a top driver for this patient's risk profile.",
                            "type": "Data-Driven"
                        })
            except Exception as e:
                # In a clinical environment, we log the failure to generate a specific recommendation
                # but allow the engine to proceed with other recommendations.
                print(f"DEBUG: Recommendation generation failed for factor {feat}: {e}")
                continue

        # Sort: High -> Moderate -> Low
        priority_map = {"High": 0, "Moderate": 1, "Low": 2}
        recommendations = sorted(recommendations, key=lambda x: priority_map.get(x['priority'], 3))

        return recommendations

if __name__ == "__main__":
    # Test
    sample_data = pd.DataFrame({
        'age': [60], 'sex': [1], 'cp': [4], 'trestbps': [170], 'chol': [280],
        'fbs': [0], 'restecg': [2], 'thalach': [130], 'exang': [1],
        'oldpeak': [2.5], 'slope': [2], 'ca': [2], 'thal': [7]
    })
    
    recommender = HeartDiseaseRecommender()
    patterns = recommender.infer_medical_patterns(sample_data)
    for p in patterns:
        print(f"[{p['severity']}] {p['label']}: {p['rationale']}")
