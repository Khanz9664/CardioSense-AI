# System Architecture: CardioSense AI

CardioSense AI is a multi-layered Clinical Decision Support System (CDSS) designed for high-performance cardiovascular risk assessment with a focus on trust, interpretability, and safety.

---

## 1. High-Level Component Interaction

The system follows a decoupled architecture where the **Core Intelligence Layer** is wrapped by a **Production API (FastAPI)** and served through a **Clinical Dashboard (Streamlit)**.

```mermaid
graph TB
    subgraph "Frontend Layer (Streamlit)"
        UI[Clinical Dashboard]
        Report[PDF Report Generator]
    end

    subgraph "Backend Layer (FastAPI)"
        API[RESTful API]
        Auth[Security Gate]
    end

    subgraph "Core Intelligence Layer (Python)"
        Predictor[HeartDiseasePredictor]
        Explainer[SHAP Explainability Engine]
        Safety[Safety & Confidence Engine]
        Simulator[Intervention Simulator]
        Recommender[Clinical Recommendation Engine]
    end

    subgraph "Data & Artifacts"
        Model[(XGBoost Model)]
        Meta[(Model Metadata)]
        Data[(UCI Patient Data)]
    end

    UI <--> API
    API --> Predictor
    UI --> Predictor
    Predictor --> Model
    Explainer --> Predictor
    Safety --> Predictor
    Simulator --> Model
    Recommender --> Explainer
    Report --> Core
```

---

## 2. The Training & Optimization Pipeline

We employ **XGBoost** as the primary engine, optimized via **Optuna** to ensure medical-grade accuracy.

```mermaid
graph LR
    A[(Raw Clinical Data)] --> B[Unified Preprocessing]
    B --> C{Optuna Meta-Learner}
    C --> D[XGBoost Hyper-Tuning]
    D --> E[Cross-Validation]
    E --> C
    C --> F[Optimized Model / .joblib]
    C --> G[Metadata & Metrics / .json]
    
    style C fill:#f9f,stroke:#333,stroke-width:2px
```

---

## 3. The Clinical Inference Flow

This sequence illustrates the path from a patient profile to a "What-If" intervention simulation.

```mermaid
sequenceDiagram
    participant C as Clinician
    participant U as UI (Streamlit)
    participant S as Safety Engine
    participant M as Model (XGBoost)
    participant X as Explainability (SHAP)
    participant I as Intervention Engine

    C->>U: Input Patient Vitals
    U->>S: Run OOD & Guardrail Checks
    S-->>U: Confidence & Safety Alerts
    U->>M: Request Risk Probability
    M-->>U: 92% Risk Score
    U->>X: Generate Driver Analysis
    X-->>U: Waterfall SHAP Plot
    C->>I: Simulate BP reduction (-20 mmHg)
    I->>M: Re-predict with modified vitals
    M-->>I: 75% Risk Score
    I-->>U: Risk Delta: -17%
```

---

## 4. Safety & Trust Framework (`src/utils/safety_engine.py`)

In medical AI, "Black Box" models are unusable. We implement three layers of trust:

1.  **Out-of-Distribution (OOD) Detection**: Compares input data against the bounds of the training set (e.g., age ranges, BP maximums).
2.  **Clinical Guardrails**: Hard-coded medical rules that can signal a "Hypertensive Crisis" even if the AI doesn't detect heart disease.
3.  **Confidence Mapping**: A statistical derivation of model certainty, labeled as **High**, **Moderate**, or **Low**.

---

## 5. Explainability Layer (`src/explainability/`)

We use **SHAP (SHapley Additive exPlanations)** to ensure every prediction is explainable.
- **Local Explanations**: Waterfall plots showing exactly how each vital contributed to a specific patient's risk.
- **Global Explanations**: Summary plots showing the most important features across the entire population (e.g., `ca`, `oldpeak`, `thalach`).

---

## 6. Project Blueprint (Source Code Organization)

- `src/models/`: Training and real-time inference wrappers.
- `src/explainability/`: Logic for SHAP values and visualization.
- `src/simulation/`: The "What-If" engine for risk reduction projections.
- `src/recommendation/`: Pattern-based medical advice generation.
- `src/utils/`: Safety engines and report orchestration.
