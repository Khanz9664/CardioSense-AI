import pandas as pd
import joblib
import json
import os
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
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
    
    model = XGBClassifier(**params)
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    return score

def train_model(X: pd.DataFrame, y: pd.Series, tune=True):
    """
    Trains an XGBoost classifier with optional Optuna hyperparameter tuning.
    Calculates advanced metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Unified Preprocessing
    preprocessor = HeartDiseasePreprocessor()
    preprocessor.fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    best_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': 42,
        'eval_metric': 'logloss'
    }

    if tune:
        print("\n--- Starting Hyperparameter Optimization with Optuna ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)
        best_params.update(study.best_params)
        print(f"Best trial accuracy: {study.best_value:.4f}")
        print(f"Best parameters: {best_params}")

    # Initialize and fit final model
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Advanced Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate median healthy profile (Low Risk)
    healthy_df = X[y == 0]
    median_healthy = healthy_df.median().to_dict()
    
    # Save a small reference set for global SHAP
    X_ref = X_train.sample(min(100, len(X_train)), random_state=42)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "best_params": best_params,
        "healthy_baseline": median_healthy,
        "X_reference": X_ref,
        "preprocessor": preprocessor # Will be handled by the specialized save function
    }
    
    print(f"\nFinal Model Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC Score: {metrics['roc_auc']:.4f}")
    
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
