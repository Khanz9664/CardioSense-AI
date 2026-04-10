import pytest
import pandas as pd
import joblib
import os
from src.explainability.explainer import HeartDiseaseExplainer

@pytest.fixture
def sample_patient():
    return pd.DataFrame({
        'age': [50],
        'sex': [1],
        'cp': [2],
        'trestbps': [120],
        'chol': [230],
        'fbs': [0],
        'restecg': [1],
        'thalach': [150],
        'exang': [0],
        'oldpeak': [1.0],
        'slope': [2],
        'ca': [0],
        'thal': [3]
    })

@pytest.fixture
def explainer():
    MODEL_PATH = "models/heart_disease_model.joblib"
    PREPROCESSOR_PATH = "models/preprocessor.joblib"
    X_REF_PATH = "models/X_reference.joblib"
    
    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)):
        pytest.skip("Model artifacts not found. Skipping explainer tests.")
        
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return HeartDiseaseExplainer(model, preprocessor=preprocessor, X_reference_path=X_REF_PATH)

def test_shap_explanation_generation(explainer, sample_patient):
    """Verify local SHAP values are calculated and formatted correctly."""
    shap_values = explainer.get_explanations(sample_patient)
    assert shap_values is not None
    assert len(shap_values.values) == 1
    # Check if feature names are present
    assert hasattr(shap_values, 'feature_names')
    assert len(shap_values.feature_names) > 0

def test_lime_explanation_generation(explainer, sample_patient):
    """Verify LIME generates local explanations when reference data is present."""
    if explainer.lime_explainer is None:
        pytest.skip("LIME explainer not initialized (missing X_reference).")
        
    lime_exp = explainer.get_lime_explanation(sample_patient)
    assert lime_exp is not None
    # LIME explanation should have list of features as tuples
    assert len(lime_exp.as_list()) > 0

def test_reasoning_summary_formatting(explainer, sample_patient):
    """Ensure clinical reasoning summary is human-readable and contains Top 3 factors."""
    shap_values = explainer.get_explanations(sample_patient)
    summary = explainer.get_reasoning_summary(shap_values)
    
    assert "Top 3 factors" in summary
    assert "(" in summary # Indicates direction (increasing/decreasing)
    assert summary.count(",") >= 2 # Should list at least 3 factors

def test_patient_comparison_against_baseline(explainer, sample_patient):
    """Verify relative patient comparison against healthy medians."""
    healthy_baseline = {
        'age': 45, 'trestbps': 110, 'chol': 180, 'thalach': 165, 'oldpeak': 0.0
    }
    comparison_df = explainer.get_patient_comparison(sample_patient, healthy_baseline)
    
    assert isinstance(comparison_df, pd.DataFrame)
    assert not comparison_df.empty
    assert "Metric" in comparison_df.columns
    assert "Delta" in comparison_df.columns

def test_global_explanation_plot(explainer):
    """Verify global summary plot generation (returns a figure)."""
    if explainer.X_reference is None:
        pytest.skip("Reference data missing. Skipping global explanation test.")
        
    fig = explainer.get_global_explanation()
    assert fig is not None
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
