import pytest
import pandas as pd
import numpy as np
from src.simulation.engine import HeartDiseaseSimulator

class MockModel:
    """Mock model that returns risk proportional to thalach (for testing logic)"""
    def predict_proba(self, X):
        # probability is thalach / 220 (roughly)
        prob = np.clip(X['thalach'].iloc[0] / 220.0, 1e-5, 1-1e-5)
        return np.array([[1 - prob, prob]])

@pytest.fixture
def simulator():
    return HeartDiseaseSimulator(MockModel())

def test_physiological_bounds_age_hr_cap(simulator):
    # Age 80 -> Max HR cap = 220 - 80 = 140
    data = pd.DataFrame({'age': [80]})
    updates = {'thalach': 160}
    bounded = simulator.apply_physiological_bounds(data, updates)
    
    assert bounded['thalach'] == 140 # Capped at 220-80

def test_physiological_bounds_physical_limits(simulator):
    # Test clinical hard bounds (e.g. BP can't be 300)
    data = pd.DataFrame({'age': [50]})
    updates = {'trestbps': 300, 'chol': 50, 'oldpeak': -5.0}
    bounded = simulator.apply_physiological_bounds(data, updates)
    
    assert 'trestbps' in bounded and bounded['trestbps'] <= 200
    assert 'chol' in bounded and bounded['chol'] >= 100
    assert 'oldpeak' in bounded and bounded['oldpeak'] >= 0.0

def test_trajectory_generation(simulator):
    # Test that trajectory has correct steps and decreasing risk
    start_df = pd.DataFrame({
        'age': [30], 'sex': [1], 'cp': [1], 'trestbps': [180], 'chol': [300], 'fbs': [0],
        'restecg': [0], 'thalach': [190], 'exang': [0], 'oldpeak': [3.0], 'ca': [0], 'thal': [3]
    })
    targets = {'thalach': 110} # In mock model, thalach reduction reduces risk
    
    trajectory = simulator.simulate_trajectory(start_df, targets, steps=5)
    
    assert len(trajectory) == 6
    assert trajectory[5]['prob'] < trajectory[0]['prob']

def test_optimization_solver_convergence(simulator):
    # Test that solver finds a path to reach a target risk
    start_df = pd.DataFrame({
        'age': [50], 'sex': [1], 'cp': [1], 'trestbps': [160], 'chol': [280], 'fbs': [0],
        'restecg': [0], 'thalach': [180], 'exang': [0], 'oldpeak': [2.0], 'ca': [0], 'thal': [3]
    })
    
    # Target risk 50%. thalach=180 -> prob=180/220 = 0.81
    # Optimization should decrease risk because clinical target for thalach is 180 (no reduction)
    # Wait, simulator.targets['thalach'] = 180. If start is 180, it won't optimize thalach.
    # I should change simulator.targets for the test or use a different feature.
    simulator.targets['thalach'] = 110.0 
    
    result = simulator.optimize_target_risk(start_df, target_risk_pct=50.0)
    
    assert result['final_prob'] < 0.81
    assert result['status'] in ["Success", "Max iterations reached (Partial)"]
