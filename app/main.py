import streamlit as st
import pandas as pd
import hashlib
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

import importlib

# Force reload internal modules to bypass Streamlit's old sys.modules cache
import src.explainability.explainer
import src.utils.safety_engine
import src.simulation.engine
import src.utils.report_generator
importlib.reload(src.explainability.explainer)
importlib.reload(src.utils.safety_engine)
importlib.reload(src.simulation.engine)
importlib.reload(src.utils.report_generator)

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
def load_clinical_vFINAL_wow():
    X_REF_PATH = os.path.join(BASE_DIR, "models", "X_reference.joblib")
    predictor = HeartDiseasePredictor(MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH)
    # Explainer for SHAP/LIME
    explainer = HeartDiseaseExplainer(
        predictor.model, 
        preprocessor=predictor.preprocessor, 
        X_reference_path=X_REF_PATH
    )
    simulator = HeartDiseaseSimulator(predictor.model, preprocessor=predictor.preprocessor)
    recommender = HeartDiseaseRecommender()
    safety_engine = HeartDiseaseSafetyEngine()
    
    metadata = None
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
            
    return predictor, explainer, simulator, recommender, safety_engine, metadata

try:
    predictor, explainer, simulator, recommender, safety_engine, metadata = load_clinical_vFINAL_wow()
except Exception as e:
    st.error(f"System Offline: {e}. Please ensure the training pipeline has been completed.")
    st.stop()

# --- PDF UTILITIES ---
def get_report_generator(audit_hash):
    return ClinicalReportGenerator(logo_path=LOGO_PATH, audit_hash=audit_hash)

def create_radar_chart_pdf(input_df, opt_result, simulator):
    try:
        r_feats = ['trestbps', 'chol', 'thalach', 'oldpeak']
        labels = ['BP', 'Chol', 'MaxHR', 'ST-Dep']
        
        def norm_radar(val, feat):
            # Same normalization logic as UI for consistency
            b = simulator.hard_bounds[feat]
            i = simulator.targets[feat]
            bad = b[1] if feat != 'thalach' else b[0]
            return abs(val - i) / abs(bad - i)

        c_vals = [norm_radar(input_df[f].iloc[0], f) for f in r_feats]
        o_vals = [norm_radar(opt_result['optimized_vitals'][f], f) for f in r_feats]
        
        # Close the loop
        c_vals.append(c_vals[0])
        o_vals.append(o_vals[0])
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        labels_loop = labels + [labels[0]]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, c_vals, color='#0056b3', linewidth=2, label='Current Patient')
        ax.fill(angles, c_vals, color='#0056b3', alpha=0.25)
        ax.plot(angles, o_vals, color='#28a745', linewidth=2, label='Clinical Target')
        ax.fill(angles, o_vals, color='#28a745', alpha=0.25)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_ylim(0, 1.2)
        ax.set_yticklabels([])
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plt.savefig(tmp.name, dpi=120, bbox_inches='tight', transparent=False)
            plt.close(fig)
            return tmp.name
    except Exception as e:
        st.error(f"Radar capture failed: {e}")
        return None

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
    st.info(f"Medical Grade AI | v{metadata.get('version', '1.0.0') if metadata else 'N/A'} Production")
    st.sidebar.markdown("---")
    st.sidebar.caption(f" **Clinical Engine**: v{metadata.get('version', '1.0.0') if metadata else 'N/A'}")
    st.sidebar.caption(f" **Audit Hash**: {hashlib.md5(str(metadata).encode()).hexdigest()[:8].upper() if metadata else 'N/A'}")

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
clinical_overrides = safety_engine.get_clinical_overrides(input_df)
clinical_assessment = safety_engine.get_clinical_assessment(input_df)
confidence = safety_engine.calculate_confidence(probability[0][1])

# Handle Overrides
display_risk_val = risk_val
if clinical_overrides:
    display_risk_val = max(risk_val, 92.5) # Minimum high-risk floor for clinical crisis

if safety_warnings or clinical_overrides:
    with st.expander(" System Integrity Alerts", expanded=True):
        for o in clinical_overrides:
            st.error(f"**CRITICAL OVERRIDE:** {o['reason']}")
        for w in safety_warnings:
            st.warning(f"**OUT-OF-DISTRIBUTION:** {w}")
        st.info("The prediction below incorporates clinical safety overrides due to acute patient vitals.")

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
        value = display_risk_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Current Risk Pulse", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#ff4b4b" if display_risk_val > 50 else "#28a745"},
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
    status_class = "risk-high" if (prediction[0] == 1 or clinical_overrides) else "risk-low"
    status_text = "NEGATIVE (Low Risk)" if (prediction[0] == 0 and not clinical_overrides) else "POSITIVE (High Risk)"
    status_color = "#28a745" if (prediction[0] == 0 and not clinical_overrides) else "#ff4b4b"
    
    st.markdown(f"""
    <div class="metric-card {status_class}">
        <h3 style="margin-top:0;">Patient Risk Summary</h3>
        <p style="font-size: 1.2rem;">Assessment Result: <b style="color:{status_color};">{status_text}</b></p>
        <p>The model predicts a <b>{display_risk_val:.1f}%</b> probability of underlying heart disease based on the current clinical profile.</p>
        <div style="margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">
            <div style="font-size: 0.9rem; font-weight: 700; color: #495057; margin-bottom: 5px;">
                Entropy Confidence: <span style="color: {confidence['color']};">{confidence['level']}</span> ({confidence['score']}%)
            </div>
            <div style="font-size: 0.8rem; color: #6c757d;">
                {confidence['description']} (Binary Entropy: {confidence['entropy']})
            </div>
        </div>
        <hr>
        <small>Reliability Score: High (90.16% Acc) | Last Optimized: April 2026</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Compute SHAP Values Early
    shap_vals = explainer.get_explanations(input_df)
    
    # Auto-generate Model Reasoning Summary
    reasoning_summary = explainer.get_reasoning_summary(shap_vals)
    
    st.info(f" **Model Reasoning Summary:** {reasoning_summary}")
    
    # PDF Download Tooling
    recs = recommender.generate_prioritized_recommendations(input_df, probability[0][1], shap_vals)
    
    # Prepare SHAP Waterfall Plot for PDF
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig_tmp, ax_tmp = plt.subplots(figsize=(10, 6))
        # Use Transformed Data to match SHAP value dimensions
        transformed_sample = predictor.preprocessor.transform(input_df)
        clean_exp_pdf = shap.Explanation(
            values=shap_vals.values[0],
            base_values=float(shap_vals.base_values[0]),
            data=transformed_sample.iloc[0].values,
            feature_names=transformed_sample.columns.tolist()
        )
        # Switch to Bar plot for local attribution (identical data, higher stability)
        shap.plots.bar(clean_exp_pdf, max_display=14, show=False)
        plt.savefig(tmpfile.name, dpi=150, bbox_inches='tight')
        plt.close(fig_tmp)
        shap_plot_path = tmpfile.name

    # Check for simulation state
    opt_full = st.session_state.get('last_opt_full', None)
    confidence = safety_engine.calculate_confidence(probability[0][1])
    audit_hash = hashlib.md5(str(metadata).encode()).hexdigest()[:12].upper()
    
    radar_plot_path = None
    if opt_full:
        radar_plot_path = create_radar_chart_pdf(input_df, opt_full, simulator)

    generator = get_report_generator(audit_hash)
    pdf_output = generator.generate_report(
        input_df=input_df,
        prediction=prediction,
        probability=probability,
        shap_plot_path=shap_plot_path,
        recs=recs,
        reasoning=reasoning_summary,
        overrides=clinical_overrides,
        assessment=clinical_assessment,
        opt_results=opt_full,
        roadmap=st.session_state.get('roadmap', None),
        observations=clinical_notes,
        confidence=confidence,
        radar_plot_path=radar_plot_path
    )
    
    st.download_button(
        label="📥 Download Clinical PDF Report",
        data=bytes(pdf_output),
        file_name=f"cardio_report_{datetime.date.today()}.pdf",
        mime="application/pdf",
        width='stretch'
    )
    
    # Cleanup temp files
    if os.path.exists(shap_plot_path):
        os.remove(shap_plot_path)
    if radar_plot_path and os.path.exists(radar_plot_path):
        os.remove(radar_plot_path)

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
        st.subheader("Local Interpretability (SHAP & LIME)")
        
        st.write("**SHAP Waterfall Base Contribution**")
        # Use Transformed Data to match SHAP value dimensions
        transformed_sample = predictor.preprocessor.transform(input_df)
        clean_exp_ui = shap.Explanation(
            values=shap_vals.values[0],
            base_values=float(shap_vals.base_values[0]),
            data=transformed_sample.iloc[0].values,
            feature_names=transformed_sample.columns.tolist()
        )
        fig_wf, ax_wf = plt.subplots(figsize=(8, 6))
        # Switch to Bar plot for local attribution (identical data, higher stability)
        shap.plots.bar(clean_exp_ui, max_display=14, show=False)
        st.pyplot(plt.gcf())
        st.caption("SHAP Bar chart showing absolute contribution (log-odds) of each vital to the prediction.")
        
        st.markdown("---")
        st.write("**LIME Linear Surrogate Perturbations**")
        with st.spinner("Generating local LIME perturbation model..."):
            lime_exp = explainer.get_lime_explanation(input_df)
            if lime_exp:
                fig_lime = lime_exp.as_pyplot_figure()
                fig_lime.set_size_inches(8, 6)
                st.pyplot(fig_lime)
                st.caption("LIME visualization displaying linear surrogate weights driving local probability.")
            else:
                st.warning("LIME explainer not initialized.")

    with diag_col2:
        st.subheader("Diagnostic Assessment")
        if clinical_assessment:
            ca_df = pd.DataFrame(clinical_assessment)
            st.table(ca_df)
        else:
            st.success("No acute clinical guideline violations detected.")
            
        st.markdown("---")
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

    # --- WOW FEATURE: ADVANCED CLINICAL OPTIMIZATION ---
    st.markdown("---")
    opt_col1, opt_col2 = st.columns([1.2, 1])
    
    # 1. RADAR VISUALIZATION
    with opt_col1:
        st.subheader(" Optimization Engine")
        st.write("Calculates the **Least Effort Path** using clinical cost weights.")
        target_risk = st.select_slider("Target Clinical Risk (%)", options=[5, 10, 15, 20, 25, 30, 40, 50], value=20)
        
        if st.button(" Generate Treatment Roadmap", width='stretch'):
            with st.spinner("Calculating..."):
                opt_result = simulator.optimize_target_risk(input_df, target_risk)
                st.session_state['last_opt_full'] = opt_result
                st.session_state['roadmap'] = simulator.get_intervention_sequence(input_df, opt_result['optimized_vitals'])
        
        if 'last_opt_full' in st.session_state:
            res = st.session_state['last_opt_full']
            def norm_radar(val, feat):
                b = simulator.hard_bounds[feat]
                i = simulator.targets[feat]
                bad = b[1] if feat != 'thalach' else b[0]
                return abs(val - i) / abs(bad - i)

            r_feats = ['trestbps', 'chol', 'thalach', 'oldpeak']
            labels = ['BP', 'Chol', 'MaxHR', 'ST-Dep']
            c_vals = [norm_radar(input_df[f].iloc[0], f) for f in r_feats]
            o_vals = [norm_radar(res['optimized_vitals'][f], f) for f in r_feats]
            
            c_vals.append(c_vals[0]); o_vals.append(o_vals[0]); labels.append(labels[0])

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=c_vals, theta=labels, fill='toself', name='Current', line_color='#0056b3'))
            fig_radar.add_trace(go.Scatterpolar(r=o_vals, theta=labels, fill='toself', name='Target', line_color='#28a745'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1.2], showticklabels=False)), 
                                   height=350, margin=dict(l=30, r=30, t=30, b=30), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_radar, width='stretch')

    # 2. ROADMAP STEPS
    with opt_col2:
        if 'roadmap' in st.session_state:
            st.subheader(" Treatment Roadmap")
            for i, step in enumerate(st.session_state['roadmap']):
                effort = " Easy" if step['effort_score'] <= 1.0 else " Moderate" if step['effort_score'] <= 2.0 else " High"
                with st.expander(f"STEP {i+1}: {step['factor']}", expanded=(i<2)):
                    st.write(f"**Action:** {step['action']}")
                    st.write(f"**Goal:** {step['impact']}")
                    st.write(f"**Effort:** {effort}")
            
            diff_p = (probability[0][1] - st.session_state['last_opt_full']['final_prob']) * 100
            st.success(f"Potential Risk Reduction: **-{diff_p:.1f}%**")
            st.caption("Following this prioritized logic leads to a projected risk of **{:.1f}%**.".format(st.session_state['last_opt_full']['final_prob']*100))
        else:
            st.info(" Run the Optimization Solver to generate a patient-specific roadmap.")

with tab3:
    st.subheader("Population-Level Feature Importance")
    st.write("dataset-wide summary show which factors most strongly drive cardiovascular risk across all patients.")
    
    global_fig = explainer.get_global_explanation()
    if global_fig:
        col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
        with col_g2:
            st.pyplot(global_fig, width='stretch')
    else:
        st.warning("Global insights require pre-trained reference data.")
        
    if metadata and 'feature_analysis' in metadata:
        fa = metadata['feature_analysis']
        st.markdown("---")
        st.subheader("Algorithmic Feature Analysis")
        
        fa_col1, fa_col2, fa_col3 = st.columns([1.1, 1.1, 1])
        with fa_col1:
            st.write("**Native Feature Importance**")
            # Show all features as requested
            imp_df = pd.DataFrame(list(fa['importance'].items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=True)
            fig_imp, ax_imp = plt.subplots(figsize=(9, 8))
            ax_imp.barh(imp_df['Feature'], imp_df['Importance'], color='#339af0')
            ax_imp.set_xlabel("Relative Importance", fontsize=9)
            ax_imp.tick_params(axis='y', labelsize=8)
            ax_imp.tick_params(axis='x', labelsize=8)
            st.pyplot(fig_imp)
            
        with fa_col2:
            st.write("**Permutation Importance**")
            if 'permutation_importance' in fa:
                # Show all features as requested
                perm_df = pd.DataFrame(list(fa['permutation_importance'].items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=True)
                fig_perm, ax_perm = plt.subplots(figsize=(9, 8))
                ax_perm.barh(perm_df['Feature'], perm_df['Importance'], color='#fcc419')
                ax_perm.set_xlabel("Mean AUC Drop", fontsize=9)
                ax_perm.tick_params(axis='y', labelsize=8)
                ax_perm.tick_params(axis='x', labelsize=8)
                st.pyplot(fig_perm)
            else:
                st.info("Permutation importance unavailable.")
                
        with fa_col3:
            st.write("**Explanation Consistency**")
            if 'explanation_consistency' in fa:
                spearman = fa['explanation_consistency'].get('spearman_correlation', 0)
                st.metric("SHAP vs Native Consistency", f"{spearman:.1%}")
                if spearman > 0.8:
                    st.success("High consistency. SHAP attributions robustly align with internal model logic.")
                else:
                    st.warning("Moderate consistency. Base model exhibits complex structure deviating from linear SHAP attribution.")
            
            st.write("**Variance Inflation Factor**")
            vif_df = pd.DataFrame(list(fa['vif'].items()), columns=['Feature', 'VIF Score']).sort_values(by='VIF Score', ascending=False)
            try:
                styled_vif = vif_df.style.map(lambda x: "background-color: #ffebee; color: #ff4b4b; font-weight: bold;" if isinstance(x, (int, float)) and x > 5 else "")
            except AttributeError:
                styled_vif = vif_df.style.applymap(lambda x: "background-color: #ffebee; color: #ff4b4b; font-weight: bold;" if isinstance(x, (int, float)) and x > 5 else "")
            st.dataframe(styled_vif, width='stretch')
            st.caption("VIF > 5 indicates multicollinearity.")
            
        st.write("**Feature Correlation Heatmap**")
        corr_df = pd.DataFrame(fa['correlation'])
        # Significant size expansion for readability
        fig_corr, ax_corr = plt.subplots(figsize=(12, 7)) 
        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax_corr, 
                    cbar_kws={'label': 'Pearson Correlation'},
                    annot_kws={"size": 5}, linewidths=0.2)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        
        col_c1, col_c2, col_c3 = st.columns([1, 4, 1]) # Expand display column
        with col_c2:
            st.pyplot(fig_corr, width='stretch')

with tab4:
    col_acc, col_auc, col_ver, col_prauc = st.columns(4)
    col_acc.metric("Clinical Accuracy", f"{metadata.get('accuracy', 0)*100:.2f}%")
    col_auc.metric("ROC-AUC Score", f"{metadata.get('roc_auc', 0):.4f}")
    col_prauc.metric("PR-AUC Score", f"{metadata.get('pr_auc', 0):.4f}")
    col_ver.metric("Model Version", "XGB-O.1.2")
    
    col_prec, col_rec, col_f1, col_brier = st.columns(4)
    col_prec.metric("Precision", f"{metadata.get('precision', 0):.4f}")
    col_rec.metric("Recall (Sensitivity)", f"{metadata.get('recall', 0):.4f}")
    col_f1.metric("F1 Score", f"{metadata.get('f1', 0):.4f}")
    col_brier.metric("Brier Score", f"{metadata.get('brier_score', 0):.4f}", help="Lower is better (closer to 0 is perfectly calibrated)")
    
    st.markdown("---")
    cm_col, cal_col = st.columns([1, 1])
    
    with cm_col:
        st.write("**Validation Confusion Matrix**")
        cm = np.array(metadata['confusion_matrix'])
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax_cm,
                    xticklabels=['Negative', 'Positive'], yticklabels=['Actual Neg', 'Actual Pos'])
        st.pyplot(fig_cm)
    
    with cal_col:
        if 'calibration_curve' in metadata:
            st.write("**Model Calibration Curve**")
            cal = metadata['calibration_curve']
            fig_cal, ax_cal = plt.subplots(figsize=(6, 4))
            ax_cal.plot(cal['prob_pred'], cal['prob_true'], marker='o', linewidth=2, color='#ff4b4b', label='XGBoost')
            ax_cal.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
            ax_cal.set_xlabel('Mean Predicted Probability')
            ax_cal.set_ylabel('Fraction of Positives')
            ax_cal.legend(loc="lower right")
            st.pyplot(fig_cal)
        else:
            st.write("**Hyperparameter Blueprint (Optuna Optimized)**")
            st.json(metadata.get('best_params', {}))
    
    if 'calibration_curve' in metadata:
        with st.expander("View Hyperparameter Blueprint"):
            st.json(metadata.get('best_params', {}))
            
    if metadata and 'bias_fairness' in metadata:
        st.markdown("---")
        st.subheader("Bias & Fairness Assessment")
        st.write("Ensuring robust equitable performance across patient subgroups.")
        
        bias = metadata['bias_fairness']
        bias_df = pd.DataFrame(bias).T
        # Select columns to display
        st.dataframe(bias_df[['count', 'accuracy', 'recall', 'f1']].style.format({
            'accuracy': '{:.1%}',
            'recall': '{:.1%}',
            'f1': '{:.1%}'
        }), width='stretch')
        st.info("**Equitable Care Parity Check:** Strong clinical models must maintain uniformly high Recall (True Positive Rate) across marginalized or vulnerable subgroups (e.g. Female and Senior populations) to avoid disparate treatment impact.")
