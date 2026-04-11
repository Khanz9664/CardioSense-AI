# Development Guide: CardioSense AI (v2.4.0)

This guide contains the necessary steps to set up, develop, and train CardioSense AI.

---

## 1. Local Setup

### System Isolation (Venv)
```bash
# Clone the repository
cd CardioSense-AI

# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate
# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Running the Clinical Pipeline

### Core Training (`main.py`)
To train the XGBoost model and optimize it using Optuna:
```bash
python main.py
```
This script will:
1. Load raw data from `data/raw/`.
2. Run the **Robust Preprocessing Pipeline** (`ColumnTransformer` with `StandardScaler` and `OneHotEncoder`).
3. Use **Optuna** to find the best hyperparameters.
4. Save the artifacts:
   - `models/heart_disease_model.joblib`: The optimized XGBoost learner.
   - `models/preprocessor.joblib`: The fitted Scikit-Learn preprocessing pipeline.
   - `models/model_metadata.json`: The clinical metrics and versioning data.

### Launching the Dashboard (Streamlit)
```bash
streamlit run app/main.py
```

### Starting the Production API (FastAPI)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
This gateway utilizes a modern **Lifespan** context manager for robust initialization of clinical artifacts and model telemetry.

---

## 3. Testing Strategy (Clinical & API)

Tests are located in the `tests/` directory and use `pytest`.

```bash
# Run the full clinical validation suite
PYTHONPATH=. .venv/bin/pytest tests/ --cov=src --cov-report=term-missing
```

- **`tests/test_safety_engine.py`**: Validates ACC/AHA hypertension guardrails and clinical overrides.
- **`tests/test_explainer.py`**: (NEW) Verifies SHAP and LIME interpretability for varied patient risk profiles.
- **`tests/test_monitoring_engine.py`**: (NEW) Validates Data Drift and Performance Auditing logic.
- **`tests/test_report_generator.py`**: (NEW) Ensures deterministic medical PDF generation of clinical results.
- **`tests/test_simulator.py`**: Verifies the "Least Effort Path" optimization logic.
- **`tests/test_api_v2.py`**: Verifies production headers, model versioning, and standardized error responses.

---

## 4. Security & Compliance Workflows

To ensure the clinical safety of the codebase, we maintain a 100% security pass rate.

### Static Analysis (Bandit)
Scans the codebase for insecure Python patterns.
```bash
# Run full security audit
bandit -r . -x ./tests,./venv,./.venv
```

### Dependency Audit (Safety)
Scans `requirements.txt` for known vulnerabilities in third-party libraries.
```bash
# Run dependency audit
safety check -r requirements.txt
```

---

## 5. Clinical Auditability

- **Audit Hash**: The dashboard displays a unique SHA-256 hash of the loaded model metadata. This allows clinicians to verify that the decision support engine has not been altered since its last validated training run.
- **Access Logs**: The system records every inference request with its associated probability and clinical reasoning in `logs/cardiosense.log`.

---

## 6. CI/CD & Automated Clinical Pipelines

Every push to the `main` branch triggers an automated **Clinical Decision Guardrail Pipeline** via GitHub Actions:

1.  **Job 1: Linting**: Ensures code quality and clinical-grade standards using `flake8`.
2.  **Job 2: Clinical Testing**: Automates the full `pytest` suite across the Safety, API, Monitoring, and Simulator modules.
3.  **Job 3: Model Ingest Audit**: Verifies that new clinical data patterns correctly traverse the `ColumnTransformer` preprocessing layer.
4.  **Job 4: Security Audit**: Runs `bandit` and `safety` scans. The pipeline will fail if any High or Medium severity vulnerabilities are detected.
5.  **Job 5: Docker Build**: Packages the FastAPI inference gateway into a production-ready container (`Dockerfile`) to ensure deployment portability.

---

---
 
 ## 7. Model Hyperparameter Blueprint (v2.4.0)
 
 The current production model was optimized using Optuna (100 trials) to achieve a clinical-grade accuracy of **88.52%** ($N=303$).
 
 ```json
 {
   "n_estimators": 80,
   "max_depth": 12,
   "learning_rate": 0.14765286433763314,
   "scale_pos_weight": 1.1685393258426966,
   "random_state": 42,
   "eval_metric": "logloss",
   "subsample": 0.7827180847167622,
   "colsample_bytree": 0.7749553588022996,
   "min_child_weight": 9,
   "gamma": 2.192444713662289
 }
 ```
 
 ---
 
 ## 8. Dependencies

- **Modeling**: `xgboost`, `scikit-learn`, `optuna`, `joblib`.
- **Explainability**: `shap`, `matplotlib`, `seaborn`.
- **API/App**: `fastapi`, `uvicorn`, `streamlit`, `plotly`.
- **Reporting**: `fpdf`.
