# Development Guide: CardioSense AI

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

## 3. Testing Strategy

Tests are located in the `tests/` directory and use `pytest`.

```bash
# Run all clinical validation tests
pytest tests/
```

- **Safety Tests**: Validate that clinical guardrails (e.g., BP > 180) are functioning.
- **Inference Tests**: Ensure that the predictor returns the expected risk scores for known cases.
- **OOD Tests**: Verify that the safety engine correctly flags out-of-distribution patients.

---

## 4. Dependencies

- **Modeling**: `xgboost`, `scikit-learn`, `optuna`, `joblib`.
- **Explainability**: `shap`, `matplotlib`, `seaborn`.
- **API/App**: `fastapi`, `uvicorn`, `streamlit`, `plotly`.
- **Reporting**: `fpdf`.
