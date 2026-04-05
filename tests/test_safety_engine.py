import pytest
import pandas as pd
import numpy as np
from src.utils.safety_engine import HeartDiseaseSafetyEngine

@pytest.fixture
def safety_engine():
    return HeartDiseaseSafetyEngine()

@pytest.fixture
def base_patient():
    return pd.DataFrame({
        'age': [55], 'sex': [1], 'cp': [1], 'trestbps': [120], 'chol': [180],
        'fbs': [0], 'restecg': [0], 'thalach': [150], 'exang': [0],
        'oldpeak': [0.0], 'slope': [1], 'ca': [0], 'thal': [3]
    }, index=[0])

def test_hypertensive_crisis_override(safety_engine, base_patient):
    # Test that BP >= 180 triggers CRITICAL override
    crisis_patient = base_patient.copy()
    crisis_patient['trestbps'] = 185
    
    overrides = safety_engine.get_clinical_overrides(crisis_patient)
    assert len(overrides) > 0
    assert any("Hypertensive Crisis" in r['reason'] for r in overrides)
    assert all(r['forced_risk'] == 1 for r in overrides)

def test_metabolic_crisis_override(safety_engine, base_patient):
    # Test that Multivessel disease (ca=2) triggers override
    vessel_patient = base_patient.copy()
    vessel_patient['ca'] = 2
    
    overrides = safety_engine.get_clinical_overrides(vessel_patient)
    assert len(overrides) > 0
    assert any("Multivessel Disease" in r['reason'] for r in overrides)

def test_aha_bp_stages(safety_engine, base_patient):
    # 1. Normal BP
    stage_normal = safety_engine.get_clinical_assessment(base_patient)
    # The implementation only returns abnormalities, so Normal BP results in empty assessment for BP
    assert not any(a['factor'] == "Blood Pressure" for a in stage_normal)
    
    # 2. Stage 1 Hypertension (130-139)
    s1_patient = base_patient.copy()
    s1_patient['trestbps'] = 135
    stage_s1 = safety_engine.get_clinical_assessment(s1_patient)
    bp_row = next(a for a in stage_s1 if a['factor'] == "Blood Pressure")
    assert "Stage 1 Hypertension" in bp_row['status']
    assert bp_row['severity'] == "MODERATE"

    # 3. Stage 2 Hypertension (>=140)
    s2_patient = base_patient.copy()
    s2_patient['trestbps'] = 150
    stage_s2 = safety_engine.get_clinical_assessment(s2_patient)
    bp_row = next(a for a in stage_s2 if a['factor'] == "Blood Pressure")
    assert "Stage 2 Hypertension" in bp_row['status']
    assert bp_row['severity'] == "HIGH"

def test_confidence_logic(safety_engine):
    # confidence returns a dict with 'score', 'level', etc.
    res_05 = safety_engine.calculate_confidence(0.5)
    assert res_05['score'] == 0.0
    assert res_05['level'] == "LOW"
    
    res_10 = safety_engine.calculate_confidence(1.0)
    assert res_10['score'] == 100.0
    assert res_10['level'] == "HIGH"

def test_out_of_distribution_bounds(safety_engine, base_patient):
    # Test detection of clinically impossible vitals
    ood_patient = base_patient.copy()
    ood_patient['trestbps'] = 250 
    
    warnings = safety_engine.check_out_of_distribution(ood_patient)
    assert len(warnings) > 1 or (len(warnings) == 1 and "TRESTBPS" in warnings[0])
    assert any("TRESTBPS" in w and "ABOVE" in w for w in warnings)
