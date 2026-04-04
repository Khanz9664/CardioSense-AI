import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import plotly.graph_objects as go
from fpdf import FPDF
import io
import datetime
import tempfile
import matplotlib
matplotlib.use('Agg') # Ensure headless compatibility
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.default'] = 'regular' # Prevent math-text parsing errors

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.predict import HeartDiseasePredictor
from src.explainability.explainer import HeartDiseaseExplainer
from src.simulation.engine import HeartDiseaseSimulator
from src.recommendation.engine import HeartDiseaseRecommender
from src.utils.report_generator import ClinicalReportGenerator
from src.utils.safety_engine import HeartDiseaseSafetyEngine

# Page Config
st.set_page_config(page_title="CardioSense AI | Clinical Decision Support", layout="wide", page_icon="❤️")

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8f9fa;
    }
    
    /* Card Container */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    
    .risk-high { border-top: 5px solid #ff4b4b; }
    .risk-low { border-top: 5px solid #28a745; }
    
    /* Recommendations Styling */
    .priority-badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
    }
    
    .high-priority { background-color: #ff4b4b; color: white; }
    .moderate-priority { background-color: #fcc419; color: black; }
    .low-priority { background-color: #339af0; color: white; }

    .rec-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dee2e6;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Navbar styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD ASSETS ---
# Resolve absolute paths for robust asset loading
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_PATH = os.path.join(BASE_DIR, "app", "assets", "logo.png")
MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_disease_model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")

@st.cache_resource
def load_components():
    predictor = HeartDiseasePredictor(MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH)
    explainer = HeartDiseaseExplainer(predictor.model)
    simulator = HeartDiseaseSimulator(predictor.model)
    recommender = HeartDiseaseRecommender()
    safety_engine = HeartDiseaseSafetyEngine()
    
    metadata = None
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
            
    return predictor, explainer, simulator, recommender, safety_engine, metadata

try:
    predictor, explainer, simulator, recommender, safety_engine, metadata = load_components()
except Exception as e:
    st.error(f"System Offline: {e}. Please ensure the training pipeline has been completed.")
    st.stop()

# --- PDF UTILITY ---
def get_report_generator():
    return ClinicalReportGenerator(logo_path=LOGO_PATH)

# --- SIDEBAR INPUTS ---
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=100)
    st.title("Patient Inputs")
    
    def user_input_features():
        age = st.slider("Patient Age", 20, 100, 50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], 
                          format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}[x])
        trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
        chol = st.slider("Serum Cholesterol", 100, 600, 230)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        restecg = st.selectbox("Resting ECG", [0, 1, 2], index=0, 
                               format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}[x])
        thalach = st.slider("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1], help="Pain during exertion",
                             format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.slider("ST Depression (Exercise)", 0.0, 6.0, 1.0)
        slope = st.selectbox("ST Slope", [1, 2, 3], index=1,
                             format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x])
        ca = st.slider("Major Vessels (Fluoroscopy)", 0, 3, 0)
        thal = st.selectbox("Thalassemia Score", [3, 6, 7], index=0,
                            format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x])
        
        return pd.DataFrame({
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }, index=[0])

    input_df = user_input_features()
    st.markdown("---")
    clinical_notes = st.text_area("Clinician's Observations", placeholder="Enter specific patient notes or follow-up instructions here...", height=100)
    st.info("Medical Grade AI | v1.2 Production")

# --- MAIN DASHBOARD ---
# 1. Safety & Trust Header
st.markdown("""
<div style="background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #0056b3;">
    <h5 style="margin: 0; color: #0056b3;"> Clinical Decision Support Mode</h5>
    <p style="margin: 0; font-size: 0.85rem; color: #495057;">
        <b>NOTICE AND DISCLAIMER:</b> This system is designed for <b>decision support only</b> and does not constitute medical advice or a formal diagnosis. 
        Always consult with a qualified healthcare professional. Model predictions are based on statistical patterns and may not account for individual patient complexities.
    </p>
</div>
""", unsafe_allow_html=True)

# 2. Prediction Core
prediction, probability = predictor.predict(input_df)
risk_val = probability[0][1] * 100

# 3. Safety Check
safety_warnings = safety_engine.check_out_of_distribution(input_df)
guardrails = safety_engine.get_clinical_guardrails(input_df)
confidence = safety_engine.calculate_confidence(probability[0][1])

if safety_warnings or guardrails:
    with st.expander(" System Integrity Alerts", expanded=True):
        for g in guardrails:
            st.error(f"**CRITICAL GUARDRAIL:** {g}")
        for w in safety_warnings:
            st.warning(f"**OUT-OF-DISTRIBUTION:** {w}")
        st.info("The prediction below may be unreliable as the patient's vitals fall outside the model's primary training data.")

# Dashboard Header with Logo
title_col1, title_col2 = st.columns([0.1, 0.9])
with title_col1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=80)
with title_col2:
    st.title("CardioSense AI Dashboard")
st.markdown("Precision Cardiovascular Risk Assessment and Decision Support")

# Top Layout: Gauge + Summary Card
top_col1, top_col2 = st.columns([1, 1.5])

with top_col1:
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Current Risk Pulse", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#ff4b4b" if risk_val > 50 else "#28a745"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#e8f5e9'},
                {'range': [30, 70], 'color': '#fff3e0'},
                {'range': [70, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, width='stretch')

with top_col2:
    status_class = "risk-high" if prediction[0] == 1 else "risk-low"
    status_text = "NEGATIVE (Low Risk)" if prediction[0] == 0 else "POSITIVE (High Risk)"
    status_color = "#28a745" if prediction[0] == 0 else "#ff4b4b"
    
    st.markdown(f"""
    <div class="metric-card {status_class}">
        <h3 style="margin-top:0;">Patient Risk Summary</h3>
        <p style="font-size: 1.2rem;">Assessment Result: <b style="color:{status_color};">{status_text}</b></p>
        <p>The model predicts a <b>{risk_val:.1f}%</b> probability of underlying heart disease based on the current clinical profile.</p>
        <div style="margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">
            <div style="font-size: 0.9rem; font-weight: 700; color: #495057; margin-bottom: 5px;">
                Prediction Confidence: <span style="color: {confidence['color']};">{confidence['level']}</span> ({confidence['score']}%)
            </div>
            <div style="font-size: 0.8rem; color: #6c757d;">
                {confidence['description']}
            </div>
        </div>
        <hr>
        <small>Reliability Score: High (90.16% Acc) | Last Optimized: April 2026</small>
    </div>
    """, unsafe_allow_html=True)
    
    # PDF Download Tooling
    import tempfile
    
    shap_vals = explainer.get_explanations(input_df)
    recs = recommender.generate_prioritized_recommendations(input_df, probability[0][1], shap_vals)
    
    # Prepare SHAP Waterfall Plot for PDF
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig_tmp, ax_tmp = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_vals[0], show=False)
        # Use bbox_inches='tight' in savefig instead of tight_layout() 
        # to avoid Matplotlib's mathtext parse exceptions on SHAP labels
        plt.savefig(tmpfile.name, dpi=150, bbox_inches='tight')
        plt.close(fig_tmp)
        shap_plot_path = tmpfile.name

    # Check for simulation state
    sim_results = st.session_state.get('last_sim', None)

    generator = get_report_generator()
    pdf_output = generator.generate_report(
        input_df=input_df,
        prediction=prediction,
        probability=probability,
        shap_plot_path=shap_plot_path,
        recs=recs,
        sim_results=sim_results,
        observations=clinical_notes,
        confidence=confidence,
        safety_warnings=safety_warnings
    )
    
    st.download_button(
        label="📥 Download Clinical PDF Report",
        data=bytes(pdf_output),
        file_name=f"cardio_report_{datetime.date.today()}.pdf",
        mime="application/pdf",
        width='stretch'
    )
    
    # Cleanup temp file
    if os.path.exists(shap_plot_path):
        os.remove(shap_plot_path)

st.markdown("---")

# TABS FOR DEEP DIVE
tab1, tab2, tab3, tab4 = st.tabs([
    " Diagnosis & Benchmarks", 
    " Intervention Simulator", 
    " Global Insights", 
    " System Integrity"
])

with tab1:
    diag_col1, diag_col2 = st.columns([1.2, 1])
    
    with diag_col1:
        st.subheader("Clinical Driver Analysis (SHAP)")
        fig_wf, ax_wf = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(shap_vals[0], show=False)
        st.pyplot(plt.gcf())
        st.caption("Waterfall plot showing contribution of each vital to the increased risk (red) or decreased risk (blue).")

    with diag_col2:
        st.subheader("Clinical Benchmarking")
        if metadata and 'healthy_baseline' in metadata:
            comp_df = explainer.get_patient_comparison(input_df, metadata['healthy_baseline'])
            st.dataframe(comp_df.set_index('Metric'), width='stretch')
            st.info("Benchmarked against the 'Healthy Median' - the ideal cardiovascular profile derived from the population dataset.")
        
        st.markdown("---")
        st.subheader("Targeted Action Plan")
        if recs:
            for r in recs:
                badge_type = r['priority'].lower() + "-priority"
                st.markdown(f"""
                <div class="rec-card">
                    <span class="priority-badge {badge_type}">{r['priority']} PRIORITY</span>
                    <div style="font-weight: 700; font-size: 1.1rem;">{r['title']}</div>
                    <div style="font-style: italic; color: #666;">{r['rationale']}</div>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.subheader("Multi-Factor Intervention Dashboard")
    st.write("Simulate hospital interventions or lifestyle changes to project future risk reduction.")
    
    sim_input_col, sim_output_col = st.columns([1, 1.3])
    
    with sim_input_col:
        s_chol = st.slider("Target Cholesterol", 100, 400, int(input_df['chol'].iloc[0]))
        s_bp = st.slider("Target Systolic BP", 80, 200, int(input_df['trestbps'].iloc[0]))
        s_thal = st.slider("Target Exercise Heart Rate", 60, 220, int(input_df['thalach'].iloc[0]))
        s_peak = st.slider("Target ST Depression Recovery", 0.0, 6.0, float(input_df['oldpeak'].iloc[0]))

    with sim_output_col:
        # Run Multi-Simulation
        updates = {'chol': s_chol,'trestbps': s_bp,'thalach': s_thal,'oldpeak': s_peak}
        multi_sim = simulator.simulate_multi_change(input_df, updates)
        st.session_state['last_sim'] = multi_sim # Save for report
        
        s_risk = multi_sim['new_prob'] * 100
        o_risk = multi_sim['original_prob'] * 100
        delta = s_risk - o_risk
        
        # Big Metric
        st.markdown(f"""
        <div style="text-align:center; padding:2rem; background:#f1f3f5; border-radius:15px;">
            <div style="font-size: 1.5rem; color:#495057;">Projected Outcome</div>
            <div style="font-size: 4rem; font-weight:800; color:{'#28a745' if delta < 0 else '#ff4b4b'};">
                {s_risk:.1f}%
            </div>
            <div style="font-size: 1.2rem; background:white; display:inline-block; padding: 5px 15px; border-radius:20px; border:1px solid #dee2e6;">
                Risk Delta: {delta:+.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if delta < -10:
            st.balloons()
            st.success(" **Optimal Intervention Strategy**: This combination of vitals represents a major clinical recovery path.")

with tab3:
    st.subheader("Population-Level Feature Importance")
    st.write("dataset-wide summary show which factors most strongly drive cardiovascular risk across all patients.")
    
    global_fig = explainer.get_global_explanation()
    if global_fig:
        st.pyplot(global_fig)
    else:
        st.warning("Global insights require pre-trained reference data.")

with tab4:
    col_acc, col_auc, col_ver = st.columns(3)
    col_acc.metric("Clinical Accuracy", f"{metadata['accuracy']*100:.2f}%")
    col_auc.metric("ROC-AUC Score", f"{metadata['roc_auc']:.4f}")
    col_ver.metric("Model Version", "XGB-O.1.2")
    
    st.markdown("---")
    cm_col, hyp_col = st.columns([1, 1])
    
    with cm_col:
        st.write("**Validation Confusion Matrix**")
        cm = np.array(metadata['confusion_matrix'])
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax_cm,
                    xticklabels=['Negative', 'Positive'], yticklabels=['Actual Neg', 'Actual Pos'])
        st.pyplot(fig_cm)
    
    with hyp_col:
        st.write("**Hyperparameter Blueprint (Optuna Optimized)**")
        st.json(metadata['best_params'])
