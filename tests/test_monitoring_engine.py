import pytest
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
from src.monitoring.engine import MonitoringEngine
from src.monitoring.logger import MonitoringLogger

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test_monitoring.db"
    return str(db_path)

@pytest.fixture
def temp_reference(tmp_path):
    ref_path = tmp_path / "X_reference.joblib"
    df = pd.DataFrame({
        'age': np.random.randint(20, 80, 100),
        'trestbps': np.random.randint(100, 180, 100),
        'chol': np.random.randint(150, 300, 100),
        'thalach': np.random.randint(100, 200, 100),
        'oldpeak': np.random.uniform(0, 4, 100),
        'probability': np.random.uniform(0, 1, 100)
    })
    joblib.dump(df, ref_path)
    return str(ref_path)

@pytest.fixture
def monitoring_engine(temp_db, temp_reference, tmp_path):
    # Metadata mockup
    meta_path = tmp_path / "model_metadata.json"
    import json
    with open(meta_path, 'w') as f:
        json.dump({"recall": 0.85}, f)
        
    engine = MonitoringEngine(
        reference_path=temp_reference,
        metadata_path=str(meta_path),
        db_path=temp_db
    )
    # Override report dir to tmp
    engine.report_dir = str(tmp_path / "reports")
    os.makedirs(engine.report_dir, exist_ok=True)
    return engine

def test_drift_analysis_with_insufficient_data(monitoring_engine):
    """Verify drift analysis handles empty production logs gracefully."""
    results = monitoring_engine.run_drift_analysis()
    assert results['status'] == "insufficient_data"
    assert results['drift_detected'] is False

def test_performance_audit_logic(monitoring_engine, temp_db):
    """Verify recall calculation using clinician feedback."""
    logger = MonitoringLogger(temp_db)
    
    # Log 20 simulated inferences
    for i in range(20):
        # 15 Correct, 5 Incorrect (Recall test)
        target = 1 if i < 15 else 0
        pred = 1 if i < 12 else 0 # 12 TP out of 15 Positives = 0.8 Recall
        
        logger.log_prediction(
            request_id=f"REQ-{i}",
            input_df=pd.DataFrame({'age': [50]}),
            prediction=pred,
            probability=0.7,
            model_version="1.0.0"
        )
        # Log feedback for all
        logger.log_feedback(f"REQ-{i}", target)
        
    audit = monitoring_engine.run_performance_audit()
    assert audit['status'] == "success"
    # Recall = 12 / 15 = 0.8
    assert audit['current_recall'] == 0.8
    assert audit['baseline_recall'] == 0.85
    assert audit['recall_drop'] == pytest.approx(0.05)
    assert audit['concept_drift_detected'] is False

def test_data_drift_report_generation(monitoring_engine, temp_db, temp_reference):
    """Verify Evidently report HTML is generated when data is present."""
    logger = MonitoringLogger(temp_db)
    
    # Log some data
    df_new = pd.DataFrame({
        'age': np.random.randint(20, 80, 50),
        'trestbps': np.random.randint(100, 180, 50),
        'chol': np.random.randint(150, 300, 50),
        'thalach': np.random.randint(100, 200, 50),
        'oldpeak': np.random.uniform(0, 4, 50)
    })
    
    for i, row in df_new.iterrows():
        logger.log_prediction(
            request_id=f"DR-TEST-{i}",
            input_df=pd.DataFrame([row.to_dict()]),
            prediction=1,
            probability=0.8,
            model_version="1.0.0"
        )
        
    results = monitoring_engine.run_drift_analysis(window_size=50)
    assert results['status'] == "success"
    assert "report_path" in results
    assert os.path.exists(results['report_path'])
    assert results['columns_monitored'] > 0
