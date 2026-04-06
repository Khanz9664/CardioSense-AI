import pandas as pd
import numpy as np
import uuid
import time
import random
import os
import sys
import joblib

# Add root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.monitoring.logger import MonitoringLogger
from src.monitoring.engine import MonitoringEngine

def simulate_clinical_traffic(n=50, drift=False):
    logger = MonitoringLogger()
    
    # Load training reference specifically for columns
    ref_path = "/home/shahid/Desktop/new datasets/heart disease/CardioSense-AI/models/X_reference.joblib"
    if not os.path.exists(ref_path):
        print(f"Error: Reference data not found at {ref_path}. Run training first.")
        return
    
    ref_df = joblib.load(ref_path)
    cols = ref_df.columns
    print(f"Simulating {n} clinical interactions (Drift={drift})...")
    
    # Baseline means for numeric features (to simulate drift)
    # ['num__age', 'num__trestbps', 'num__chol', 'num__thalach', 'num__oldpeak']
    
    for i in range(n):
        request_id = str(uuid.uuid4())
        
        # Pull a random sample from reference
        sample = ref_df.iloc[[random.randint(0, len(ref_df)-1)]].copy()
        
        if drift:
            # Shift age and BP higher (Systematic Drift)
            if 'num__age' in sample.columns:
                sample['num__age'] = sample['num__age'] + 0.5 # Normalized shift
            if 'num__trestbps' in sample.columns:
                sample['num__trestbps'] = sample['num__trestbps'] + 0.3
            
        # Mock prediction logic
        # If age/BP high, more likely to predict 1
        risk_score = random.random()
        if drift: risk_score += 0.2
        
        prediction = 1 if risk_score > 0.6 else 0
        probability = min(0.99, max(0.01, risk_score))
        
        logger.log_prediction(
            request_id=request_id,
            input_df=sample,
            prediction=prediction,
            probability=probability,
            model_version="2.0.0"
        )
        
        # Simulate clinician feedback for 30% of cases
        if random.random() < 0.3:
            # High correlation with prediction for low drift, lower for high drift
            accuracy = 0.9 if not drift else 0.7
            actual = prediction if random.random() < accuracy else (1 - prediction)
            logger.log_feedback(request_id, actual)
            
    print(f"Simulation of {'Drifted' if drift else 'Stable'} traffic complete.")

if __name__ == "__main__":
    # Create monitoring dir if missing
    os.makedirs("/home/shahid/Desktop/new datasets/heart disease/CardioSense-AI/data/monitoring", exist_ok=True)
    
    # 1. Simulate stable traffic
    simulate_clinical_traffic(n=60, drift=False)
    # 2. Simulate drifted traffic
    simulate_clinical_traffic(n=40, drift=True)
    
    # 3. Trigger manual analysis
    print("\nRunning Monitoring Engine analysis...")
    engine = MonitoringEngine(
        reference_path="/home/shahid/Desktop/new datasets/heart disease/CardioSense-AI/models/X_reference.joblib",
        metadata_path="/home/shahid/Desktop/new datasets/heart disease/CardioSense-AI/models/model_metadata.json",
        db_path="/home/shahid/Desktop/new datasets/heart disease/CardioSense-AI/data/monitoring/inference_history.db"
    )
    
    drift_stats = engine.run_drift_analysis(window_size=100)
    print("Drift Results Summary:", drift_stats)
    
    perf_stats = engine.run_performance_audit()
    print("Performance Audit Summary:", perf_stats)
