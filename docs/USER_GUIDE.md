# Clinical User Guide: CardioSense AI

CardioSense AI facilitates advanced cardiovascular decision-making through an interactive dashboard and automated clinical reporting.

---

## 1. Dashboard Overview

The CardioSense AI dashboard provides a comprehensive medical interface for risk assessment.

````carousel
![Patient Risk Summary](../app/assets/App_Screenshots/1.png)
<!-- slide -->
![Model Reasoning Summary](../app/assets/App_Screenshots/2.png)
<!-- slide -->
![Risk Optimization Radar](../app/assets/App_Screenshots/6.png)
````

### Patient Inputs & Risk Pulse
- **Sidebar**: Input traditional cardiovascular risk factors (Age, BP, Cholesterol, etc.).
- **Risk Pulse Gauge**: Real-time visual indicator of heart disease probability.
![Patient Inputs](../app/assets/App_Screenshots/1.png)
![Risk Pulse Gauge](../app/assets/App_Screenshots/2.png)
---

## 2. Deep Dive Modules

### Diagnosis & Benchmarks
Analyze the **underlying drivers** of the patient's risk.

![Clinical Driver Analysis](../app/assets/App_Screenshots/3.png)
![Actionable LIME Insights](../app/assets/App_Screenshots/4.png)

- **SHAP Waterfall Analysis**: Visualizes exactly how many percentage points each vital contributed to the overall risk. Red bars indicate increased risk; blue bars indicate protective factors.
- **Patient Benchmarking**: Compare your patient's vitals against the **Healthy Median**.

### Risk Optimization Engine (Least Effort Path)
Move beyond simple "What-If" analysis to an AI-driven clinical strategy.

![Intervention Simulation Dashboard](../app/assets/App_Screenshots/5.png)
![Risk Optimization & Radar](../app/assets/App_Screenshots/6.png)

1. **Strategic Optimization**: Select a "Target Risk" percentage and run the solver.
2. **Spider (Radar) Visualization**: 
   - **Blue Shape**: The patient's high-risk profile.
   - **Green Shape**: The AI-calculated "Path to Green."
   - **Center (0.0)**: Represents ideal clinical benchmarks. A shrinking shape indicates successful risk reduction.
3. **Treatment Roadmap**: A prioritized sequence of lifestyle and clinical actions (e.g., "Step 1: Blood Pressure Stabilization") ranked by their risk-reduction ROI relative to effort.

### Global Insights
Understand population-level data drivers.

![Population Importance](../app/assets/App_Screenshots/7.png)
![Feature Analysis](../app/assets/App_Screenshots/8.png)
![Correlation Heatmap](../app/assets/App_Screenshots/9.png)

- View feature importance across the entire dataset to see which factors strongest drive risk globally.

### System Integrity
Audit the engine's reliability.

![Model Metrics and Parameters](../app/assets/App_Screenshots/10.png)
![Audit & Fairness](../app/assets/App_Screenshots/11.png)

- Review **Accuracy**, **ROC-AUC**, and the **Confusion Matrix** to ensure clinical validity.

---

## 3. Interpreting the AI "Reasoning"

The SHAP Waterfall plot is the "X-Ray" of the model's decision. It decomposes the 0-100% risk probability into the specific clinical reasons for why a patient was flagged.

- **`E[f(X)]`**: The average model output (the starting baseline).
- **`f(X)`**: The final risk probability for this specific patient.
- **Red Features**: Clinical factors that pushed the risk **Higher**.
- **Blue Features**: Clinical factors that pushed the risk **Lower**.

---

## 4. Generating Clinical PDF Reports

After completing your assessment and running simulations, generate a professional report for the patient's medical file:

1.  Input clinician observations in the text field.
2.  Click **"Download Clinical PDF Report"**.
3.  The generated PDF includes:
    - Patient Risk Summary.
    - Full SHAP Waterfall Analysis.
    - Clinical Benchmarking Table.
    - Intervention Strategy projections.
    - Confidence scores and safety integrity logs.
