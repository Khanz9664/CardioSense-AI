# Development Guide: CardioSense AI (v2.1.0)

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

---

## 3. Testing Strategy (Clinical & API)

Tests are located in the `tests/` directory and use `pytest`.

```bash
# Run the full clinical validation suite
PYTHONPATH=. .venv/bin/pytest tests/
```

- **`tests/test_safety_engine.py`**: Validates ACC/AHA hypertension guardrails, clinical overrides, and entropy-based confidence.
- **`tests/test_simulator.py`**: Ensures the Risk Optimization Engine adheres to physiological bounds and converges on the "Least Effort Path."
- **`tests/test_api_v2.py`**: Verifies production headers (`X-Request-ID`), model versioning, and standardized error responses.

---

## 4. Clinical Auditability

- **Audit Hash**: The dashboard displays a unique SHA-256 hash of the loaded model metadata. This allows clinicians to verify that the decision support engine has not been altered since its last validated training run.
- **Access Logs**: The system records every inference request with its associated probability and clinical reasoning in `logs/cardiosense.log`.

---

## 5. CI/CD & Automated Clinical Pipelines

Every push to the `main` branch triggers an automated **Clinical Decision Guardrail Pipeline** via GitHub Actions:

1.  **Job 1: Linting**: Ensures code quality and clinical-grade standards using `flake8`.
2.  **Job 2: Clinical Testing**: Automates the full `pytest` suite across the Safety, API, and Simulator modules.
3.  **Job 3: Model Ingest Audit**: Verifies that new clinical data patterns correctly traverse the `ColumnTransformer` preprocessing layer.
4.  **Job 4: Docker Build**: Packages the FastAPI inference gateway into a production-ready container (`Dockerfile`) to ensure deployment portability.

---

## 6. Dependencies

- **Modeling**: `xgboost`, `scikit-learn`, `optuna`, `joblib`.
- **Explainability**: `shap`, `matplotlib`, `seaborn`.
- **API/App**: `fastapi`, `uvicorn`, `streamlit`, `plotly`.
- **Reporting**: `fpdf`.
