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
2. Run the `Unified Preprocessor`.
3. Use **Optuna** to find the best hyperparameters.
4. Save the artifacts (`.joblib`) and metadata (`.json`) to the `models/` directory.

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

## 5. Dependencies

- **Modeling**: `xgboost`, `scikit-learn`, `optuna`, `joblib`.
- **Explainability**: `shap`, `matplotlib`, `seaborn`.
- **API/App**: `fastapi`, `uvicorn`, `streamlit`, `plotly`.
- **Reporting**: `fpdf`.
