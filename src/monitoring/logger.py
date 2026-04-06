import sqlite3
import pandas as pd
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any

class MonitoringLogger:
    """
    Handles persistence of inference requests, model predictions, and clinician feedback
    to a local SQLite database for drift monitoring and performance auditing.
    """
    def __init__(self, db_path: str = "data/monitoring/inference_history.db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Creates the inference history table if it does not exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for storing inference records
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inference_logs (
                request_id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                input_data TEXT, -- JSON string of input features
                prediction INTEGER,
                probability REAL,
                actual_outcome INTEGER DEFAULT NULL, -- Ground truth provided later
                model_version TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_prediction(self, request_id: str, input_df: pd.DataFrame, prediction: int, probability: float, model_version: str):
        """Asynchronously (from API perspective) logs a prediction to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        input_json = input_df.to_json(orient='records')
        # We strip the outer brackets from Orient='records' if it's a single row
        input_json = input_json[1:-1] if input_json.startswith('[') else input_json

        cursor.execute('''
            INSERT INTO inference_logs (request_id, input_data, prediction, probability, model_version)
            VALUES (?, ?, ?, ?, ?)
        ''', (request_id, input_json, int(prediction), float(probability), model_version))
        
        conn.commit()
        conn.close()

    def log_feedback(self, request_id: str, outcome: int):
        """Updates an existing record with ground truth feedback."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE inference_logs 
            SET actual_outcome = ? 
            WHERE request_id = ?
        ''', (int(outcome), request_id))
        
        conn.commit()
        conn.close()

    def get_recent_logs(self, limit: int = 1000) -> pd.DataFrame:
        """Retrieves recent logs as a pandas DataFrame for drift analysis."""
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT * FROM inference_logs ORDER BY timestamp DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return df

        # Parse the JSON input_data back into columns
        parsed_inputs = df['input_data'].apply(lambda x: json.loads(x))
        input_df = pd.DataFrame(parsed_inputs.tolist())
        
        # Combine with metadata
        final_df = pd.concat([df[['request_id', 'timestamp', 'prediction', 'probability', 'actual_outcome', 'model_version']], input_df], axis=1)
        return final_df

if __name__ == "__main__":
    # Quick Test
    logger = MonitoringLogger()
    print("Inference logger initialized.")
