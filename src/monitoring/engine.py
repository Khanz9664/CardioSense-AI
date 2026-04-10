import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Dict, Any, Optional, Tuple
from evidently import Report
from evidently.metrics import DriftedColumnsCount
from scipy.stats import ks_2samp
from src.monitoring.logger import MonitoringLogger

class MonitoringEngine:
    """
    Orchestrates the calculation of Data, Prediction, and Concept drift using Evidently AI.
    Integrates with the local SQLite inference history for comparison against training baselines.
    """
    def __init__(self, 
                 reference_path: str = "models/X_reference.joblib",
                 metadata_path: str = "models/model_metadata.json",
                 db_path: str = "data/monitoring/inference_history.db"):
        self.reference_path = reference_path
        self.metadata_path = metadata_path
        self.logger = MonitoringLogger(db_path)
        self.report_dir = "reports/monitoring"
        os.makedirs(self.report_dir, exist_ok=True)

    def _load_reference_data(self) -> pd.DataFrame:
        """Loads the training snapshot used as the monitoring baseline."""
        if not os.path.exists(self.reference_path):
            # Fallback: Create empty DF with expected columns if reference is missing
            return pd.DataFrame()
        return joblib.load(self.reference_path)

    def run_drift_analysis(self, window_size: int = 500) -> Dict[str, Any]:
        """
        Performs Data and Prediction drift analysis comparing Production vs Baseline.
        Returns a summary dictionary of drift scores.
        """
        ref_df = self._load_reference_data()
        current_df = self.logger.get_recent_logs(limit=window_size)
        
        if ref_df.empty or current_df.empty:
            return {"status": "insufficient_data", "drift_detected": False}

        # Align columns (drop metadata columns from current_df for comparison)
        compare_cols = [c for c in ref_df.columns if c in current_df.columns]
        
        # 1. Data Drift Report (Using the robust DriftedColumnsCount metric)
        drift_report = Report([DriftedColumnsCount()])
        snapshot = drift_report.run(current_data=current_df[compare_cols], reference_data=ref_df[compare_cols])
        
        # Save HTML Report for the UI to embed
        html_path = os.path.join(self.report_dir, "data_drift.html")
        snapshot.save_html(html_path)
        
        # Extract Summary Metrics using Snapshot properties
        report_json = snapshot.dict()
        
        # In 0.7.x, the structure might be different. We'll try to find 'share_of_drifted_columns'
        # Typically it's in the results of the DataDriftPreset metrics.
        drift_share = 0.0
        dataset_drift = False
        
        try:
            # Look for DataDriftTable result
            for metric in report_json.get('metrics', []):
                if 'share_of_drifted_columns' in metric.get('result', {}):
                    drift_share = float(metric['result']['share_of_drifted_columns'])
                    dataset_drift = bool(metric['result']['dataset_drift'])
                    break
        except Exception as e:
            print(f"Warning: Could not extract drift metrics from JSON: {e}")

        # 2. Prediction Drift (Target Drift) - Manual Robust Calculation
        # We perform a Kolmogorov-Smirnov test to detect distribution shift in probabilities
        # We need a reference. For now, we'll use a uniform distribution if we don't have ref probabilities,
        # but the best reference is the training probabilities.
        # Since we're in the app, we'll compare the current window vs the total log history as a relative drift.
        
        target_drift_detected = False
        p_value = 1.0
        
        try:
            # Get full history from DB for a more stable reference
            full_df = self.logger.get_recent_logs(limit=2000)
            if not full_df.empty and len(full_df) > len(current_df):
                ref_probs = full_df['probability'].values
                cur_probs = current_df['probability'].values
                
                # KS Test (Two-sample)
                # Only run if both arrays have variance to avoid RuntimeWarning: divide by zero
                # We check std > 1e-9 for numerical stability
                if np.std(cur_probs) > 1e-9 and np.std(ref_probs) > 1e-9:
                    ks_stat, p_value = ks_2samp(cur_probs, ref_probs)
                    # Ensure p_value is standard float
                    p_value = float(p_value)
                else:
                    # If both are constant and equal, no drift. If constant and different, drift.
                    p_value = 1.0 if np.all(cur_probs == ref_probs[0]) else 0.0
                
                # p-value < 0.05 indicates drift
                target_drift_detected = bool(p_value < 0.05)
        except Exception as e:
            print(f"Warning: Manual Target Drift Test failed: {e}")

        return {
            "status": "success",
            "drift_share": round(float(drift_share), 4),
            "dataset_drift": bool(dataset_drift or target_drift_detected),
            "target_drift_p": round(float(p_value), 4),
            "columns_monitored": int(len(compare_cols)),
            "last_updated": str(current_df['timestamp'].iloc[0]) if not current_df.empty else None,
            "report_path": str(html_path)
        }

    def run_performance_audit(self) -> Dict[str, Any]:
        """
        Uses clinician feedback (Actual Outcomes) to detect Concept Drift (Performance Decay).
        """
        current_df = self.logger.get_recent_logs(limit=1000)
        # Only keep records that have an actual outcome
        auditable_df = current_df[current_df['actual_outcome'].notnull()].copy()
        
        if len(auditable_df) < 10:
            return {"status": "insufficient_feedback", "count": len(auditable_df)}

        # Map actual_outcome to 'target' for Evidently
        auditable_df['target'] = auditable_df['actual_outcome']
        
        # We need a reference classification performance. 
        # We can extract baseline metrics from model_metadata.json
        baseline_recall = 0.90 # Default high baseline
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                meta = json.load(f)
                baseline_recall = meta.get('recall', 0.90)

        # Basic manual calculation of current performance
        from sklearn.metrics import recall_score, accuracy_score, roc_auc_score
        
        curr_recall = recall_score(auditable_df['target'], auditable_df['prediction'], zero_division=0)
        curr_acc = accuracy_score(auditable_df['target'], auditable_df['prediction'])
        
        recall_drop = baseline_recall - curr_recall
        
        return {
            "status": "success",
            "current_recall": round(float(curr_recall), 4),
            "baseline_recall": round(float(baseline_recall), 4),
            "recall_drop": round(float(recall_drop), 4),
            "feedback_count": int(len(auditable_df)),
            "concept_drift_detected": bool(recall_drop > 0.10) # Alert if recall drops by > 10%
        }

if __name__ == "__main__":
    engine = MonitoringEngine()
    print("Monitoring Engine initialized.")
