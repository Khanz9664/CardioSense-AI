import pytest
import pandas as pd
import numpy as np
from src.recommendation.engine import HeartDiseaseRecommender
from unittest.mock import MagicMock

@pytest.fixture
def recommender():
    return HeartDiseaseRecommender()

@pytest.fixture
def intensive_care_patient():
    # Patient with multiple high-risk markers
    return pd.DataFrame({
        'age': [65], 'sex': [1], 'cp': [4], 'trestbps': [190], 'chol': [300],
        'fbs': [1], 'restecg': [2], 'thalach': [110], 'exang': [1],
        'oldpeak': [3.0], 'slope': [3], 'ca': [3], 'thal': [7]
    }, index=[0])

@pytest.fixture
def mock_shap_values():
    # Mocking SHAP Explainer output structure
    mock = MagicMock()
    mock.feature_names = ['num__age', 'num__trestbps', 'num__chol', 'num__thalach', 'num__oldpeak']
    # High impact on trestbps and chol
    mock.values = np.array([[0.1, 0.8, 0.7, 0.2, 0.1]])
    return mock

def test_silent_ischemia_pattern(recommender):
    # Angina (exang=1) + Asymptomatic (cp=4)
    data = pd.DataFrame({
        'age': [50], 'sex': [1], 'cp': [4], 'trestbps': [120], 'chol': [200],
        'fbs': [0], 'restecg': [0], 'thalach': [150], 'exang': [1],
        'oldpeak': [0.0], 'slope': [1], 'ca': [0], 'thal': [3]
    }, index=[0])
    
    patterns = recommender.infer_medical_patterns(data)
    assert any(p['id'] == "silent_ischemia" for p in patterns)
    assert any(p['severity'] == "High" for p in patterns)

def test_hypertension_severity_levels(recommender):
    # 1. Moderate (161-180)
    data_mod = pd.DataFrame({'trestbps': [170], 'oldpeak': [0], 'thalach': [150], 'exang': [0], 'cp': [1], 'chol': [200], 'ca': [0]}, index=[0])
    patterns_mod = recommender.infer_medical_patterns(data_mod)
    hyp_mod = next(p for p in patterns_mod if p['id'] == "hypertension")
    assert hyp_mod['severity'] == "Moderate"
    
    # 2. High (>180)
    data_high = pd.DataFrame({'trestbps': [195], 'oldpeak': [0], 'thalach': [150], 'exang': [0], 'cp': [1], 'chol': [200], 'ca': [0]}, index=[0])
    patterns_high = recommender.infer_medical_patterns(data_high)
    hyp_high = next(p for p in patterns_high if p['id'] == "hypertension")
    assert hyp_high['severity'] == "High"

def test_prioritization_logic(recommender, intensive_care_patient, mock_shap_values):
    # Ensure High priority is always before Moderate
    recommendations = recommender.generate_prioritized_recommendations(
        intensive_care_patient, probability=0.85, shap_values=mock_shap_values
    )
    
    # Check that the first elements are "High" priority
    assert recommendations[0]['priority'] == "High"
    
    # Verify that priority sorting is maintained
    priority_order = [r['priority'] for r in recommendations]
    # "High" (0) < "Moderate" (1)
    for i in range(len(priority_order) - 1):
        p1 = 0 if priority_order[i] == "High" else 1
        p2 = 0 if priority_order[i+1] == "High" else 1
        assert p1 <= p2

def test_shap_augmentation(recommender, mock_shap_values):
    # Patient with NO specific patterns but HIGH SHAP on cholesterol
    data = pd.DataFrame({
        'age': [40], 'sex': [1], 'cp': [1], 'trestbps': [120], 'chol': [200], # Pattern threshold is 260
        'fbs': [0], 'restecg': [0], 'thalach': [150], 'exang': [0],
        'oldpeak': [0.0], 'slope': [1], 'ca': [0], 'thal': [3]
    }, index=[0])
    
    # Check if SHAP-driven recommendation is added
    recs = recommender.generate_prioritized_recommendations(data, 0.5, mock_shap_values)
    assert any("Optimize Lipid Management" in r['title'] for r in recs)
    assert any(r['type'] == "Data-Driven" for r in recs)
