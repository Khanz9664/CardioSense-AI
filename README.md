# CardioSense AI: Clinical Decision Support System

<p align="center">
  <img src="app/assets/logo.png" width="200" alt="CardioSense AI Banner">
</p>

---

##  Elevator Pitch

CardioSense AI is an explainable AI-powered cardiovascular decision support system that not only predicts heart disease risk but also explains the reasoning behind predictions and simulates how lifestyle changes can reduce that risk.

Unlike traditional models, it combines:
- **High-performance ML** (XGBoost + Optuna)
- **Explainability** (SHAP)
- **Real-time simulation**
- **Clinical recommendations**

**Transforming prediction into actionable medical intelligence.**

---

##  System Narrative: The Interpretability Gap

Cardiovascular disease is the world's leading killer, yet clinical adoption of AI is hampered by the "Black Box" problem. Most models provide a risk score without an explanation, leaving clinicians unable to trust or validate the AI's "intuition."

**CardioSense AI** is a professional **Clinical Decision Support System (CDSS)** designed to bridge this gap. By combining high-performance machine learning with state-of-the-art explainability and preventive simulation, it transforms raw data into trustable, actionable medical intelligence.

---

##  Documentation Portal

For detailed technical and clinical information, please refer to the following modules:

| Module | Description |
| :--- | :--- |
| **[System Architecture](docs/ARCHITECTURE.md)** | Deep dive into pipelines, safety engines, and component interactions. |
| **[Production API Guide](docs/API_GUIDE.md)** | Full FastAPI reference, schemas, and integration examples. |
| **[Clinical User Guide](docs/USER_GUIDE.md)** | Walkthrough of the dashboard and clinical PDF report interpretation. |
| **[Development Manual](docs/DEVELOPMENT.md)** | Setup instructions, training pipelines, and testing strategy. |
| **[Clinical Data Dictionary](docs/DATA_DICTIONARY.md)** | Explanation of the 13 clinical features and medical safety thresholds. |

---

##  Quick Start

### 1. Environment Initialization
```bash
# Clone and enter directory
cd CardioSense-AI
# Create virtual environment
python -m venv .venv
source .venv/bin/activate
# Install clinical stack
pip install -r requirements.txt
```

### 2. Execution Pathways
**Run Training & Optimization Pipeline:**
```bash
python main.py
```

**Launch Clinical Diagnostic Dashboard:**
```bash
streamlit run app/main.py
```

**Launch Production API Service:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

##  Core Intelligence Modules

- **The Trust & Safety Framework**: Implements OOD detection and clinical guardrails.
- **The Explainability Engine**: Powered by SHAP for local feature-level contributions.
- **The Preventive Simulation Engine**: Moves from "Reactive" to "Preventive" through "What-If" analysis.

---

##  Technical Performance

| Metric | Score | Professional Interpretation |
| :--- | :--- | :--- |
| **Clinical Accuracy** | **90.16%** | Optimized via Optuna for high stability. |
| **ROC-AUC Score** | **0.9418** | Exceptional ability to distinguish risk from health. |
| **OOD Reliability** | **Active** | Built-in warnings for anomalous patient profiles. |

---

*Disclaimer: CardioSense AI is designed exclusively for decision assistance. It is not a replacement for independent clinical judgment by a licensed medical professional.*

<p align="center">
  <a href="https://khanz9664.github.io/portfolio">
    <img src="https://img.shields.io/badge/Portfolio-255E00?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Portfolio">
  </a>
  <a href="https://github.com/khanz9664">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://www.linkedin.com/in/shahid-ul-islam-13650998/">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href="https://www.kaggle.com/shaddy9664">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle">
  </a>
  <a href="mailto:shahid9664@gmail.com">
    <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email">
  </a>
</p>