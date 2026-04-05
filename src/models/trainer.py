import pandas as pd
import numpy as np
import joblib
import json
import os
import optuna
import shap
from scipy.stats import spearmanr
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.frozen import FrozenEstimator
from src.data.preprocessor import HeartDiseasePreprocessor

def objective(trial, X, y):
    """
    Optuna objective function for XGBoost hyperparameter tuning.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    # Handle class imbalance explicitly
    scale_pos_weight = (len(y) - y.sum()) / y.sum()
    params['scale_pos_weight'] = scale_pos_weight
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = XGBClassifier(**params)
    score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    return score

def train_model(X: pd.DataFrame, y: pd.Series, tune=True):
    """
    Trains an XGBoost classifier with optional Optuna hyperparameter tuning.
    Calculates advanced metrics.
    """
    # Stratified Train, Calibration, Test Split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    # Unified Preprocessing
    preprocessor = HeartDiseasePreprocessor()
    preprocessor.fit(X_train)
    
    # Keep raw X_test for segment-based fairness checks
    X_test_raw = X_test.copy()
    
    X_train = preprocessor.transform(X_train)
    X_cal = preprocessor.transform(X_cal)
    X_test = preprocessor.transform(X_test)
    
    # Class Imbalance Handling
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    best_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'eval_metric': 'logloss'
    }

    if tune:
        print("\n--- Starting Hyperparameter Optimization with Optuna ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)
        best_params.update(study.best_params)
        best_params['scale_pos_weight'] = scale_pos_weight # Ensure it's not overwritten
        print(f"Best trial accuracy: {study.best_value:.4f}")
        print(f"Best parameters: {best_params}")

    # Initialize and fit final base model
    base_model = XGBClassifier(**best_params)
    base_model.fit(X_train, y_train)
    
    # Apply Probability Calibration (CRITICAL for medical decision making)
    model = CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method='sigmoid')
    model.fit(X_cal, y_cal)
    
    # Advanced Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    
    # Calculate median healthy profile (Low Risk)
    healthy_df = X[y == 0]
    median_healthy = healthy_df.median().to_dict()
    
    # Save a small reference set for global SHAP
    X_ref = X_train.sample(min(100, len(X_train)), random_state=42)
    
    # --- Feature Analysis ---
    corr_matrix = X_train.corr().to_dict()
    # Compute VIF from inverse correlation matrix
    inv_corr = np.linalg.inv(X_train.corr().values)
    vif = pd.Series(np.diag(inv_corr), index=X_train.columns).to_dict()
    
    # Extract Feature Importances from XGBoost
    importances = dict(zip(X_train.columns, base_model.feature_importances_.tolist()))
    
    # --- Global Permutation Importance ---
    perm_result = permutation_importance(model, X_test, y_test, scoring='roc_auc', n_repeats=5, random_state=42)
    permutation_imp_dict = dict(zip(X_train.columns, perm_result.importances_mean.tolist()))
    
    # --- Explanation Consistency Check ---
    shap_explainer_local = shap.TreeExplainer(base_model)
    shap_vals_matrix = shap_explainer_local.shap_values(X_test)
    shap_global_importances = np.abs(shap_vals_matrix).mean(axis=0)
    
    # Spearman rank correlation between Native XGB Importance and Global SHAP Importance
    spearman_corr, _ = spearmanr(base_model.feature_importances_, shap_global_importances)
    
    # --- Bias & Fairness Assessment ---
    bias_metrics = {}
    # 1. Gender slices
    for gender_val, gender_name in [(0, "Female"), (1, "Male")]:
        mask = X_test_raw["sex"] == gender_val
        if mask.sum() > 0:
            bias_metrics[f"Gender_{gender_name}"] = {
                "count": int(mask.sum()),
                "accuracy": accuracy_score(y_test[mask], y_pred[mask]),
                "recall": recall_score(y_test[mask], y_pred[mask], zero_division=0),
                "f1": f1_score(y_test[mask], y_pred[mask], zero_division=0)
            }
            
    # 2. Age slices
    for age_group, mask in {
        "Young_LT45": X_test_raw["age"] < 45,
        "Middle_45_64": (X_test_raw["age"] >= 45) & (X_test_raw["age"] <= 64),
        "Senior_GTE65": X_test_raw["age"] >= 65
    }.items():
        if mask.sum() > 0:
            bias_metrics[f"Age_{age_group}"] = {
                "count": int(mask.sum()),
                "accuracy": accuracy_score(y_test[mask], y_pred[mask]),
                "recall": recall_score(y_test[mask], y_pred[mask], zero_division=0),
                "f1": f1_score(y_test[mask], y_pred[mask], zero_division=0)
            }
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "brier_score": brier_score_loss(y_test, y_prob),
        "calibration_curve": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist()
        },
        "feature_analysis": {
            "correlation": corr_matrix,
            "vif": vif,
            "importance": importances,
            "permutation_importance": permutation_imp_dict,
            "explanation_consistency": {"spearman_correlation": spearman_corr}
        },
        "bias_fairness": bias_metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "best_params": best_params,
        "healthy_baseline": median_healthy,
        "X_reference": X_ref,
        "preprocessor": preprocessor # Will be handled by the specialized save function
    }
    
    print(f"\nFinal Model Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}  |  PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")
    
    return model, metrics

def save_model_artifacts(model, metrics, output_dir: str):
    """
    Saves the trained model and associated metrics metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract reference and preprocessor data from metrics to save separately
    X_reference = metrics.pop("X_reference", None)
    preprocessor = metrics.pop("preprocessor", None)
    
    if X_reference is not None:
        ref_path = os.path.join(output_dir, "X_reference.joblib")
        joblib.dump(X_reference, ref_path)
    
    if preprocessor is not None:
        pre_path = os.path.join(output_dir, "preprocessor.joblib")
        preprocessor.save(pre_path)
    
    # Save model
    model_path = os.path.join(output_dir, "heart_disease_model.joblib")
    joblib.dump(model, model_path)
    
    # Save metrics metadata
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Model, preprocessor, and metadata saved to {output_dir}")

if __name__ == "__main__":
    # Test script - assuming processed data exists
    PROCESSED_DATA_PATH = "data/processed/heart_disease_cleaned.csv"
    MODEL_SAVE_PATH = "models/heart_disease_model.joblib"
    
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)
        X_data = df.drop("target", axis=1)
        y_data = df["target"]
        
        model_final, metrics_final = train_model(X_data, y_data)
        save_model_artifacts(model_final, metrics_final, "models")
    else:
        print(f"Processed data not found at {PROCESSED_DATA_PATH}. Run loader.py first.")
