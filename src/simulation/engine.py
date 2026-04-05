import pandas as pd
import numpy as np
import copy

class HeartDiseaseSimulator:
    """
    Class to simulate real-time patient data changes and assess impact on risk.
    Supports multi-variable baseline comparison, automated trajectories, and risk optimization.
    """
    def __init__(self, model, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
        # Clinical targets for modifiable risk factors
        self.targets = {
            'chol': 200.0,      # mg/dl
            'trestbps': 120.0,  # mm Hg
            'thalach': 180.0,   # Higher is generally better for fitness capacity
            'oldpeak': 0.0      # No ST depression
        }
        
        # Clinical effort weights (Clinical Cost)
        # Higher = Harder to achieve in short-term lifestyle changes
        self.costs = {
            'trestbps': 1.0,   # Baseline effort (e.g. Sodium reduction/Medication)
            'chol': 1.5,       # Moderate effort (Dietary shifts)
            'thalach': 2.0,    # High effort (Sustained aerobic conditioning)
            'oldpeak': 3.5     # Extreme effort (Surgical/Structural recovery)
        }
        
        # Clinical hard bounds (to ensure physiological realism)
        self.hard_bounds = {
            'chol': (100, 500),
            'trestbps': (80, 200),
            'thalach': (60, 202),
            'oldpeak': (0.0, 6.0)
        }

    def _get_risk_proba(self, data: pd.DataFrame):
        """
        Internal helper to get probability, applying preprocessing if available.
        """
        if self.preprocessor is not None:
            processed = self.preprocessor.transform(data)
            return self.model.predict_proba(processed)[:, 1][0]
        return self.model.predict_proba(data)[:, 1][0]

    def apply_physiological_bounds(self, current_data: pd.DataFrame, updates: dict):
        """
        Enforces medical realism on simulated inputs.
        Example: Age-adjusted HR limits.
        """
        safe_updates = copy.deepcopy(updates)
        age = current_data['age'].iloc[0]
        
        # 1. 220 - Age Rule for Max HR (Thalach)
        max_possible_hr = 220 - age
        if 'thalach' in safe_updates:
            safe_updates['thalach'] = min(safe_updates['thalach'], max_possible_hr)
            
        # 2. General hard bounds
        for feature, (min_val, max_val) in self.hard_bounds.items():
            if feature in safe_updates:
                safe_updates[feature] = np.clip(safe_updates[feature], min_val, max_val)
                
        return safe_updates

    def simulate_multi_change(self, base_data: pd.DataFrame, updates: dict):
        """
        Takes a base data point, modifies multiple features, and calculates new risk.
        Includes physiological constraint filtering.
        """
        # Apply medical realism first
        safe_updates = self.apply_physiological_bounds(base_data, updates)
        
        sim_data = base_data.copy()
        original_prob = self._get_risk_proba(base_data)
        
        # Apply all safe updates
        for feature, new_value in safe_updates.items():
            if feature in sim_data.columns:
                sim_data[feature] = new_value
        
        new_prob = self._get_risk_proba(sim_data)
        delta = new_prob - original_prob
        
        return {
            "original_prob": original_prob,
            "new_prob": new_prob,
            "delta": delta,
            "updates": safe_updates
        }

    def simulate_trajectory(self, base_data: pd.DataFrame, final_targets: dict, steps: int = 5):
        """
        Generates a step-by-step risk reduction path from current to target vitals.
        Uses linear interpolation between base and target states.
        """
        trajectory = []
        features = list(final_targets.keys())
        
        for i in range(steps + 1):
            fraction = i / steps
            current_step_updates = {}
            for feat in features:
                start_val = base_data[feat].iloc[0]
                # Scale modifiable features toward target
                if feat in final_targets:
                    end_val = final_targets[feat]
                    current_step_updates[feat] = start_val + fraction * (end_val - start_val)
            
            result = self.simulate_multi_change(base_data, current_step_updates)
            trajectory.append({
                "step": i,
                "prob": result['new_prob'],
                "updates": result['updates']
            })
            
        return trajectory

    def optimize_target_risk(self, base_data: pd.DataFrame, target_risk_pct: float, max_iterations: int = 30):
        """
        Advanced Optimization Engine: Finds the LEAST EFFORT PATH to a specific target risk level.
        Uses a weighted coordinate descent approach (Reduction per Clinical Cost).
        """
        target_prob = target_risk_pct / 100
        current_data = base_data.copy()
        current_prob = self._get_risk_proba(current_data)
        modifiable = ['chol', 'trestbps', 'thalach', 'oldpeak']
        optimized_vitals = {feat: float(base_data[feat].iloc[0]) for feat in modifiable}
        
        if current_prob <= target_prob:
            return {
                "status": "Target already reached.", 
                "final_prob": current_prob, 
                "target_reached": True,
                "optimized_vitals": optimized_vitals
            }
        
        # Track which moves were most efficient
        efficiency_history = {f: 0 for f in modifiable}
        
        # Iterative improvement using Cost-Weighted Selection
        for _ in range(max_iterations):
            best_efficiency = 0
            best_feature = None
            
            for feat in modifiable:
                temp_updates = copy.deepcopy(optimized_vitals)
                target_val = self.targets[feat]
                current_val = temp_updates[feat]
                
                # Take a 10% incremental step toward the clinical target
                step_val = (target_val - current_val) * 0.10
                if abs(step_val) < 0.01: continue
                
                temp_updates[feat] += step_val
                res = self.simulate_multi_change(base_data, temp_updates)
                reduction = current_prob - res['new_prob']
                
                # Metric: Reduction per unit of Clinical Effort (Cost)
                efficiency = reduction / self.costs[feat]
                
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_feature = feat
            
            if best_feature is None or best_efficiency < 0.0001:
                break
                
            # Commit the most efficient step
            optimized_vitals[best_feature] += (self.targets[best_feature] - optimized_vitals[best_feature]) * 0.10
            
            # Combine base data with optimized vitals for next iteration
            commit_df = base_data.copy()
            for k, v in optimized_vitals.items():
                commit_df[k] = v
            current_prob = self._get_risk_proba(commit_df)
            
            if current_prob <= target_prob:
                break
                
        return {
            "status": "Success" if current_prob <= target_prob else "Partial Target Reached",
            "optimized_vitals": optimized_vitals,
            "final_prob": current_prob,
            "target_reached": current_prob <= target_prob
        }

    def get_intervention_sequence(self, base_data, optimized_vitals):
        """
        Transforms optimized numerical targets into a prioritized sequence of clinical actions.
        """
        sequence = []
        for feat, opt_val in optimized_vitals.items():
            original = base_data[feat].iloc[0]
            diff = opt_val - original
            if abs(diff) > 0.1:
                priority = "HIGH" if self.costs[feat] <= 1.5 else "MODERATE"
                impact = "Hemodynamic Stabilization" if feat == 'trestbps' else \
                         "Lipid Management" if feat == 'chol' else \
                         "Aerobic Capacity" if feat == 'thalach' else "ST-Segment Recovery"
                
                sequence.append({
                    "factor": feat.upper(),
                    "action": f"Reduce {feat.upper()} toward {opt_val:.1f}" if diff < 0 else f"Increase {feat.upper()} toward {opt_val:.1f}",
                    "impact": impact,
                    "priority": priority,
                    "effort_score": self.costs[feat]
                })
        
        return sorted(sequence, key=lambda x: x['effort_score'])

    def generate_recommendations(self, base_data: pd.DataFrame, shap_values):
        """
        Identifies key risk drivers using SHAP and calculates the potential risk reduction.
        """
        modifiable = ['chol', 'trestbps', 'thalach', 'oldpeak']
        feature_names = base_data.columns.tolist()
        patient_shap = shap_values.values[0]
        
        recommendations = []
        original_prob = self.model.predict_proba(base_data)[:, 1][0]

        for feature in modifiable:
            idx = feature_names.index(feature)
            impact = patient_shap[idx]
            current_val = base_data[feature].iloc[0]
            target_val = self.targets[feature]

            is_suboptimal = current_val < target_val if feature == 'thalach' else current_val > target_val

            if impact > 0 or is_suboptimal:
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

        return sorted(recommendations, key=lambda x: x['new_risk'])

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
