import pytest
import pandas as pd
import sqlite3
import os
import tempfile
from src.monitoring.logger import MonitoringLogger

@pytest.fixture
def temp_db():
    # Use a temporary file for the SQLite database
    fd, path = tempfile.mkstemp()
    yield path
    os.close(fd)
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture
def logger(temp_db):
    return MonitoringLogger(db_path=temp_db)

def test_db_initialization(temp_db):
    # Ensure the database and table are created on init
    logger = MonitoringLogger(db_path=temp_db)
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='inference_logs'")
    assert cursor.fetchone() is not None
    conn.close()

def test_log_prediction_integrity(logger, temp_db):
    sample_input = pd.DataFrame({'age': [55], 'trestbps': [140]}, index=[0])
    request_id = "test-req-123"
    
    logger.log_prediction(request_id, sample_input, 1, 0.85, "1.0.0")
    
    # Retrieve and verify
    df = logger.get_recent_logs(limit=1)
    assert len(df) == 1
    assert df.iloc[0]['request_id'] == request_id
    assert df.iloc[0]['age'] == 55
    assert df.iloc[0]['probability'] == 0.85

def test_log_feedback_update(logger):
    sample_input = pd.DataFrame({'age': [55]}, index=[0])
    request_id = "test-feedback-456"
    
    # 1. Log prediction
    logger.log_prediction(request_id, sample_input, 0, 0.2, "1.1.0")
    
    # 2. Log feedback (Actual outcome = 1, e.g. false negative)
    logger.log_feedback(request_id, 1)
    
    # 3. Verify
    df = logger.get_recent_logs(limit=1)
    assert df.iloc[0]['actual_outcome'] == 1

def test_concurrent_log_logic(logger):
    # Test multiple sequential logs to ensure no locking issues in basic usage
    for i in range(5):
        logger.log_prediction(f"idx-{i}", pd.DataFrame({'age': [i]}), 0, 0.1, "1.0.0")
    
    df = logger.get_recent_logs(limit=10)
    assert len(df) >= 5
    # Verify values are correct
    assert set(df['age']).issuperset({0, 1, 2, 3, 4})

def test_json_parsing_corner_cases(logger):
    # Test input with special characters in feature names or values if any
    # (Though clinical features are usually numeric)
    special_input = pd.DataFrame({'feat_with_space': [10.5]}, index=[0])
    logger.log_prediction("special-req", special_input, 1, 0.5, "1.0.0")
    
    df = logger.get_recent_logs(limit=1)
    assert df.iloc[0]['feat_with_space'] == 10.5
