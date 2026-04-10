import pytest
import pandas as pd
import os
from src.utils.report_generator import ClinicalReportGenerator

@pytest.fixture
def sample_data():
    input_df = pd.DataFrame({
        'age': [55],
        'trestbps': [140],
        'chol': [250],
        'thalach': [145],
        'oldpeak': [1.5]
    })
    prediction = [1]
    probability = [[0.2, 0.8]]
    recs = [
        {"priority": "High", "title": "Cardiology Consultation", "rationale": "High probability of coronary issues."},
        {"priority": "Moderate", "title": "Statin Therapy", "rationale": "Elevated cholesterol levels detected."}
    ]
    assessment = [
        {"factor": "Blood Pressure", "status": "Stage 1 Hypertension", "severity": "MODERATE"},
        {"factor": "ST Depression", "status": "Ischemic response", "severity": "HIGH"}
    ]
    return input_df, prediction, probability, recs, assessment

def test_report_generation_output(sample_data):
    """Verify ClinicalReportGenerator produces a non-empty byte output (PDF content)."""
    input_df, prediction, probability, recs, assessment = sample_data
    
    # Use a dummy logo path or actual logo if it exists
    logo_path = "app/assets/logo.png"
    if not os.path.exists(logo_path):
        logo_path = None
        
    generator = ClinicalReportGenerator(logo_path=logo_path, audit_hash="TEST-HASH-123")
    
    # Mocking external assets (SHAP/Radar plots) as None for pure logic test
    pdf_bytes = generator.generate_report(
        input_df=input_df,
        prediction=prediction,
        probability=probability,
        shap_plot_path=None,
        recs=recs,
        reasoning="Patient exhibits hypertensive markers and significant ST depression.",
        overrides=None,
        assessment=assessment,
        opt_results=None,
        roadmap=None,
        observations="Follow-up in 2 weeks.",
        confidence={"level": "HIGH", "score": 0.95},
        radar_plot_path=None
    )
    
    assert pdf_bytes is not None
    assert len(pdf_bytes) > 0
    # PDF files start with %PDF
    assert pdf_bytes.startswith(b"%PDF")

def test_report_metadata_integrity():
    """Verify unique report ID and date formatting."""
    generator = ClinicalReportGenerator(audit_hash="AUDIT-XYZ")
    assert len(generator.report_id) == 8
    assert generator.audit_hash == "AUDIT-XYZ"
    assert "-" in generator.report_date # Check date format contains separator
