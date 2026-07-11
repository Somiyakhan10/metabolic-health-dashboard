"""
dashboard/app.py
================
Professional Metabolic Health Dashboard with Dark Blue Theme
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Metabolic Health Dashboard",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "models/best_model.pkl"

# Physiologically plausible RMR bounds (kcal/day)
RMR_MIN_PLAUSIBLE = 600
RMR_MAX_PLAUSIBLE = 3500

# Candidate scale factors
SCALE_CANDIDATES = [1, 10, 100, 1000, 0.1, 0.01]

# ============================================================
# DARK BLUE CSS STYLING WITH WHITE TEXT - INCLUDING SIDEBAR
# ============================================================
CUSTOM_CSS = """
<style>
    /* Dark Theme */
    .main { background-color: #0a1628; }
    .stApp { background: #0a1628; }
    
    /* Sidebar - Dark Blue with White Text */
    .css-1d391kg, .css-1lcbmhc, [data-testid="stSidebar"] {
        background: #0d1f3c !important;
    }
    .css-1d391kg .stMarkdown, .css-1d391kg p, .css-1d391kg div, .css-1d391kg span {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] div, [data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] .stCaption p {
        color: #aed6f1 !important;
    }
    
    /* Cards - White text on blue */
    .metric-card {
        background: #142c4a;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        text-align: center;
        border-left: 4px solid #2e86c1;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(46, 134, 193, 0.15);
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff !important;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #ffffff !important;
        margin-top: 0.3rem;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #aed6f1 !important;
    }
    
    /* Badges - White text */
    .badge-low {
        background: #0d3b2e;
        color: #ffffff !important;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
    }
    .badge-moderate {
        background: #3d2e0d;
        color: #ffffff !important;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
    }
    .badge-high {
        background: #3d0d0d;
        color: #ffffff !important;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
    }
    .badge-good {
        background: #0d3b2e;
        color: #ffffff !important;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
    }
    .badge-fair {
        background: #3d2e0d;
        color: #ffffff !important;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
    }
    .badge-poor {
        background: #3d0d0d;
        color: #ffffff !important;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    /* All text on blue backgrounds - WHITE */
    h1, h2, h3, h4, h5, .stTitle, .stSubheader {
        color: #ffffff !important;
    }
    label, .stMarkdown, .stText, .stCaption {
        color: #ffffff !important;
    }
    
    /* Form labels */
    .stNumberInput label, .stSelectbox label, .stTextInput label {
        color: #ffffff !important;
    }
    
    /* Buttons - White text */
    .stButton > button {
        background: #1a5276;
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
        width: 100%;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #2e86c1;
        color: #ffffff !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(46, 134, 193, 0.3);
    }
    
    /* Tabs - White text */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #0d1f3c;
        padding: 8px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #85929e !important;
        padding: 8px 20px;
        border-radius: 8px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #1a5276 !important;
        color: #ffffff !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #ffffff !important;
    }
    
    /* Inputs - White text on dark */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background: #142c4a;
        color: #ffffff !important;
        border: 1px solid #1a5276;
        border-radius: 8px;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #2e86c1;
        box-shadow: 0 0 0 2px rgba(46, 134, 193, 0.2);
    }
    .stNumberInput input {
        background: #142c4a;
        color: #ffffff !important;
    }
    .stSelectbox select {
        background: #142c4a;
        color: #ffffff !important;
    }
    .stMultiSelect [data-baseweb="select"] {
        background: #142c4a;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #1a5276, #2e86c1);
    }
    
    /* Divider */
    hr {
        border-color: #1a5276;
        margin: 2rem 0;
    }
    
    /* Info Boxes - White text */
    .stInfo {
        background: #0d1f3c;
        color: #ffffff !important;
        border-left: 4px solid #2e86c1;
    }
    .stInfo p {
        color: #ffffff !important;
    }
    .stSuccess {
        background: #0d3b2e;
        color: #ffffff !important;
        border-left: 4px solid #52be80;
    }
    .stSuccess p {
        color: #ffffff !important;
    }
    .stWarning {
        background: #3d2e0d;
        color: #ffffff !important;
        border-left: 4px solid #f39c12;
    }
    .stWarning p {
        color: #ffffff !important;
    }
    .stError {
        background: #3d0d0d;
        color: #ffffff !important;
        border-left: 4px solid #e74c3c;
    }
    .stError p {
        color: #ffffff !important;
    }
    
    /* Expander - White text */
    .streamlit-expanderHeader {
        background: #0d1f3c;
        color: #ffffff !important;
        border-radius: 8px;
    }
    .streamlit-expanderHeader p {
        color: #ffffff !important;
    }
    .streamlit-expanderContent {
        background: #142c4a;
        border-radius: 0 0 8px 8px;
    }
    .streamlit-expanderContent p, .streamlit-expanderContent div {
        color: #ffffff !important;
    }
    
    /* Dataframe - White text */
    .dataframe {
        background: #142c4a !important;
        color: #ffffff !important;
    }
    .dataframe thead tr th {
        background: #1a5276 !important;
        color: #ffffff !important;
    }
    .dataframe tbody tr td {
        background: #142c4a !important;
        color: #ffffff !important;
    }
    .dataframe tbody tr td:hover {
        background: #1a5276 !important;
    }
    
    /* Plotly Charts */
    .js-plotly-plot {
        background: transparent !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0d1f3c;
    }
    ::-webkit-scrollbar-thumb {
        background: #1a5276;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #2e86c1;
    }
    
    /* Sidebar text - all white */
    .sidebar-content {
        color: #ffffff !important;
    }
    .sidebar-content p, .sidebar-content div, .sidebar-content span {
        color: #ffffff !important;
    }
    
    /* Headers in sidebar */
    .sidebar-content h1, .sidebar-content h2, .sidebar-content h3 {
        color: #ffffff !important;
    }
    
    /* Captions */
    .stCaption, .stCaption p {
        color: #aed6f1 !important;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #ffffff !important;
    }
    
    /* Selectbox dropdown */
    .stSelectbox [data-baseweb="select"] {
        background: #142c4a !important;
        color: #ffffff !important;
    }
    .stSelectbox [data-baseweb="select"] div {
        color: #ffffff !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["name"]


def render_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Personalized Metabolic Health Assessment")
        st.caption("AI-powered RMR prediction with metabolic health scoring and exercise prescriptions. Trained on NHANES 2017-2018 data.")
    with col2:
        st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=80)
    st.divider()


def patient_input_form():
    with st.form("patient_form"):
        st.subheader("Patient Information")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Demographics**")
            age = st.number_input("Age (years)", 18, 100, 45)
            sex = st.selectbox("Sex", ["Male", "Female"])
            weight = st.number_input("Weight (kg)", 30.0, 250.0, 78.0, step=0.5)
            height = st.number_input("Height (cm)", 120.0, 220.0, 170.0, step=0.5)

        with col2:
            st.markdown("**Anthropometrics**")
            waist = st.number_input("Waist circumference (cm)", 50.0, 200.0, 95.0, step=0.5)
            sbp = st.number_input("Systolic BP (mmHg)", 80.0, 220.0, 120.0, step=1.0)
            dbp = st.number_input("Diastolic BP (mmHg)", 40.0, 140.0, 75.0, step=1.0)

        with col3:
            st.markdown("**Laboratory Values**")
            glucose = st.number_input("Fasting glucose (mg/dL)", 50.0, 400.0, 95.0, step=1.0)
            hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.4, step=0.1)
            hdl = st.number_input("HDL cholesterol (mg/dL)", 10.0, 150.0, 50.0, step=1.0)
            triglycerides = st.number_input("Triglycerides (mg/dL)", 20.0, 1000.0, 120.0, step=1.0)

        submitted = st.form_submit_button("Generate Comprehensive Metabolic Report", use_container_width=True)

    if not submitted:
        return None

    bmi = weight / ((height / 100) ** 2)

    if bmi < 18.5:
        bmi_cat = "Underweight"
    elif bmi < 25:
        bmi_cat = "Normal"
    elif bmi < 30:
        bmi_cat = "Overweight"
    else:
        bmi_cat = "Obese"

    inputs = pd.DataFrame([{
        "RIDAGEYR": age,
        "RIAGENDR": 1 if sex == "Male" else 2,
        "BMXWT": weight,
        "BMXHT": height,
        "BMXBMI": bmi,
        "BMXWAIST": waist,
        "BPXSY1": sbp,
        "BPXDI1": dbp,
        "LBXGLU": glucose,
        "LBXGH": hba1c,
        "LBDHDD": hdl,
        "LBXTR": triglycerides,
        "BMI_Category": bmi_cat,
        "Waist_to_Height_Ratio": waist / height,
        "BMI_x_Age": bmi * age,
    }])

    return inputs


# ============================================================
# ALIGN FEATURES
# ============================================================
def align_features(inputs, model):
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        current_features = list(inputs.columns)

        aligned_df = inputs.copy()

        for feat in expected_features:
            if feat not in current_features:
                if feat == "HOMA_IR":
                    aligned_df[feat] = (aligned_df["LBXGLU"] * 10.0) / 405.0
                elif feat == "Metabolic_Syndrome_Flag":
                    aligned_df[feat] = 0
                elif feat == "Diabetes_Status":
                    aligned_df[feat] = "Unknown"
                elif feat == "Physical_Activity_Category":
                    aligned_df[feat] = "Unknown"
                elif feat.startswith("BMI_Category_"):
                    cat = aligned_df["BMI_Category"].iloc[0]
                    aligned_df[feat] = 1 if feat == f"BMI_Category_{cat}" else 0
                else:
                    aligned_df[feat] = 0

        aligned_df = aligned_df[expected_features]
        return aligned_df

    return inputs


# ============================================================
# SCALE AUTO-DETECTION AND CORRECTION
# ============================================================
def detect_and_correct_scale(raw_prediction: float, formula_estimate: float, model_name: str):
    is_linear_regression = "linear" in (model_name or "").lower() and "regression" in (model_name or "").lower()

    if is_linear_regression and raw_prediction > 5000:
        corrected = raw_prediction / 10.0
        note = (
            f"Detected likely target-scaling bug for model '{model_name}': raw prediction "
            f"({raw_prediction:.0f}) was divided by 10 to get {corrected:.0f} kcal/day."
        )
        return corrected, 10.0, note

    if not formula_estimate or formula_estimate <= 0:
        return raw_prediction, 1.0, None

    best_factor = 1.0
    best_score = None
    for factor in SCALE_CANDIDATES:
        candidate = raw_prediction / factor
        if RMR_MIN_PLAUSIBLE <= candidate <= RMR_MAX_PLAUSIBLE:
            ratio = candidate / formula_estimate
            score = abs(np.log(ratio))
            if best_score is None or score < best_score:
                best_score = score
                best_factor = factor

    if best_score is not None and best_factor != 1.0:
        corrected = raw_prediction / best_factor
        note = (
            f"Auto-detected a scale factor of {best_factor:g}x on the raw model output "
            f"({raw_prediction:.0f} -> {corrected:.0f} kcal/day) based on proximity to the "
            "formula-based estimate."
        )
        return corrected, best_factor, note

    return raw_prediction, 1.0, None


# ============================================================
# RMR SANITY GUARD
# ============================================================
def get_safe_rmr(candidate_prediction: float, formula_estimate: float):
    ratio = candidate_prediction / formula_estimate if formula_estimate else np.inf
    out_of_range = not (RMR_MIN_PLAUSIBLE <= candidate_prediction <= RMR_MAX_PLAUSIBLE)
    scale_mismatch = not (0.5 <= ratio <= 2.0)

    if out_of_range or scale_mismatch:
        note = (
            f"Model output ({candidate_prediction:.0f} kcal/day) is still "
            f"{'outside the plausible physiological range' if out_of_range else f'{ratio:.1f}x the formula-based estimate'} "
            "after scale correction. Falling back to the Mifflin-St Jeor estimate for display."
        )
        return formula_estimate, note, True

    return candidate_prediction, None, False


# ============================================================
# COMPUTE METRICS
# ============================================================
def compute_advanced_metrics(inputs):
    bmi = float(inputs["BMXBMI"].iloc[0])
    age = float(inputs["RIDAGEYR"].iloc[0])
    waist = float(inputs["BMXWAIST"].iloc[0])
    height = float(inputs["BMXHT"].iloc[0])
    whtr = waist / height if height > 0 else 0.5
    sbp = float(inputs["BPXSY1"].iloc[0])
    dbp = float(inputs["BPXDI1"].iloc[0])
    glucose = float(inputs["LBXGLU"].iloc[0])
    hdl = float(inputs["LBDHDD"].iloc[0])
    trig = float(inputs["LBXTR"].iloc[0])

    sex = int(inputs["RIAGENDR"].iloc[0])
    weight = float(inputs["BMXWT"].iloc[0])

    # Mifflin-St Jeor
    if sex == 1:
        rmr_est = 10.0 * weight + 6.25 * height - 5.0 * age + 5.0
    else:
        rmr_est = 10.0 * weight + 6.25 * height - 5.0 * age - 161.0

    # Metabolic Health Score (0-100)
    score = 0

    if bmi < 25:
        score += 25
    elif bmi < 30:
        score += 18
    elif bmi < 35:
        score += 10
    else:
        score += 5

    if whtr < 0.5:
        score += 20
    elif whtr < 0.6:
        score += 14
    elif whtr < 0.7:
        score += 8
    else:
        score += 4

    if sbp < 120 and dbp < 80:
        score += 20
    elif sbp < 130 and dbp < 85:
        score += 14
    else:
        score += 6

    if glucose < 100:
        score += 20
    elif glucose < 126:
        score += 12
    else:
        score += 5

    if hdl > 40 and trig < 150:
        score += 15
    elif hdl > 35 and trig < 200:
        score += 9
    else:
        score += 4

    score = min(100, max(0, score))

    # Metabolic Age
    avg_rmr_by_age = {20: 1600, 30: 1550, 40: 1490, 50: 1430, 60: 1360, 70: 1280, 80: 1200}
    ages = sorted(avg_rmr_by_age.keys())
    avg_values = [avg_rmr_by_age[a] for a in ages]
    closest_idx = np.argmin(np.abs(np.array(avg_values) - rmr_est))
    metabolic_age = ages[closest_idx]

    # Exercise Prescription
    if score >= 70:
        exercise = {
            "Focus": "Maintenance",
            "Type": "Mix of cardio and resistance training",
            "Intensity": "Moderate-Vigorous",
            "Minutes_Week": 150,
            "Resistance": "3x/week",
            "Notes": "Maintain current activity level"
        }
        status = "Good"
        badge = "badge-good"
    elif score >= 50:
        exercise = {
            "Focus": "Weight Management",
            "Type": "Brisk walking and bodyweight exercises",
            "Intensity": "Moderate",
            "Minutes_Week": 180,
            "Resistance": "2-3x/week",
            "Notes": "Gradually increase intensity"
        }
        status = "Fair"
        badge = "badge-fair"
    else:
        if bmi > 30:
            exercise = {
                "Focus": "Weight Loss",
                "Type": "Low-impact cardio and strength training",
                "Intensity": "Low-Moderate",
                "Minutes_Week": 200,
                "Resistance": "3x/week",
                "Notes": "Consult physician before starting"
            }
        else:
            exercise = {
                "Focus": "Metabolic Health",
                "Type": "Interval training and resistance",
                "Intensity": "Moderate-Vigorous",
                "Minutes_Week": 160,
                "Resistance": "3-4x/week",
                "Notes": "Focus on post-meal activity"
            }
        status = "Needs Improvement"
        badge = "badge-poor"

    if age > 60:
        exercise["Notes"] += " | Include balance exercises"
    if age > 75:
        exercise["Notes"] += " | Prioritize safety, avoid high-impact"

    # 10-Year Risk
    risk = 0
    if bmi > 30:
        risk += 2
    elif bmi > 25:
        risk += 1
    if score < 50:
        risk += 2
    elif score < 70:
        risk += 1
    if age > 50:
        risk += 1
    if age > 65:
        risk += 2
    if rmr_est < 1400:
        risk += 1
    if rmr_est < 1200:
        risk += 2

    if risk <= 2:
        risk_category = "Low"
        risk_percentage = 5 + risk * 2
    elif risk <= 5:
        risk_category = "Moderate"
        risk_percentage = 15 + (risk - 3) * 4
    else:
        risk_category = "High"
        risk_percentage = 30 + (risk - 6) * 5

    risk_percentage = min(50, risk_percentage)

    return {
        "Metabolic_Health_Score": score,
        "Metabolic_Age": metabolic_age,
        "Actual_Age": age,
        "Status": status,
        "Status_Badge": badge,
        "RMR_Estimated": rmr_est,
        "Exercise": exercise,
        "Risk_Category": risk_category,
        "Risk_Percentage": risk_percentage,
        "BMI": bmi,
        "Waist_to_Height_Ratio": whtr,
        "SBP": sbp,
        "DBP": dbp,
        "Glucose": glucose,
        "HDL": hdl,
        "Triglycerides": trig
    }


# ============================================================
# PLOT FUNCTIONS
# ============================================================
def create_rmr_comparison_plot(prediction, avg_rmr=1500):
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(['Your RMR', 'Population Average'],
                  [prediction, avg_rmr],
                  color=['#2e86c1', '#1a5276'])
    ax.axhline(y=prediction, color='#5dade2', linestyle='--', alpha=0.8)
    ax.set_ylabel('RMR (kcal/day)', color='#ffffff')
    ax.set_title('RMR Comparison', color='#ffffff')
    ax.tick_params(colors='#ffffff')
    ax.grid(axis='y', alpha=0.2, color='#1a5276')
    ax.set_facecolor('#0a1628')
    fig.patch.set_facecolor('#0a1628')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
               f'{height:.0f}', ha='center', va='bottom', fontweight='bold', color='#ffffff')
    plt.tight_layout()
    return fig


def create_metabolic_health_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Metabolic Health Score", 'font': {'color': '#ffffff'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'color': '#ffffff'}},
            'bar': {'color': "#2e86c1"},
            'steps': [
                {'range': [0, 50], 'color': "#3d0d0d"},
                {'range': [50, 70], 'color': "#3d2e0d"},
                {'range': [70, 100], 'color': "#0d3b2e"}
            ],
            'threshold': {
                'line': {'color': "#e74c3c", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=300,
        paper_bgcolor='#0a1628',
        font={'color': '#ffffff'}
    )
    return fig


def create_component_radar(metrics):
    components = {
        'BMI': 100 - min(metrics['BMI'] / 30 * 100, 100),
        'Waist/Height': 100 - min(metrics['Waist_to_Height_Ratio'] / 0.7 * 100, 100),
        'Blood Pressure': 100 - min(((metrics['SBP'] - 80) / 60) * 100, 100),
        'Glucose': 100 - min((metrics['Glucose'] - 70) / 80 * 100, 100),
        'Lipids': 100 - min(((metrics['Triglycerides'] - 50) / 200) * 100, 100)
    }

    df_radar = pd.DataFrame({
        'Component': list(components.keys()),
        'Score': list(components.values())
    })

    fig = px.line_polar(df_radar, r='Score', theta='Component',
                        line_close=True,
                        title="Metabolic Component Health",
                        range_r=[0, 100])
    fig.update_traces(fill='toself', fillcolor='rgba(46, 134, 193, 0.3)',
                      line_color='#2e86c1')
    fig.update_layout(
        height=350,
        paper_bgcolor='#0a1628',
        font={'color': '#ffffff'},
        polar=dict(
            bgcolor='#0a1628',
            radialaxis=dict(gridcolor='#1a5276', tickfont=dict(color='#ffffff')),
            angularaxis=dict(gridcolor='#1a5276', tickfont=dict(color='#ffffff'))
        )
    )
    return fig


def create_weekly_schedule(exercise):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    activity = ['Cardio'] * 7

    if exercise['Resistance'] == "3x/week":
        activity[0] = 'Cardio + Resistance'
        activity[2] = 'Cardio + Resistance'
        activity[4] = 'Cardio + Resistance'
    elif exercise['Resistance'] == "2-3x/week":
        activity[0] = 'Cardio + Resistance'
        activity[3] = 'Cardio + Resistance'
    elif exercise['Resistance'] == "3-4x/week":
        activity[0] = 'Cardio + Resistance'
        activity[2] = 'Cardio + Resistance'
        activity[4] = 'Cardio + Resistance'
        activity[5] = 'Cardio + Resistance'

    colors = {'Cardio': '#1a5276', 'Cardio + Resistance': '#2e86c1', 'Rest': '#0d1f3c'}

    fig, ax = plt.subplots(figsize=(10, 2))
    for i, (day, act) in enumerate(zip(days, activity)):
        ax.bar(i, 1, color=colors.get(act, '#95a5a6'), edgecolor='#0a1628', linewidth=2)
        ax.text(i, 0.5, f'{day}\n{act[:8]}', ha='center', va='center', fontsize=8, fontweight='bold', color='#ffffff')
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Weekly Activity Plan', fontsize=12, fontweight='bold', color='#ffffff')
    ax.set_facecolor('#0a1628')
    fig.patch.set_facecolor('#0a1628')
    plt.tight_layout()
    return fig


def create_risk_gauge(risk_percentage):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percentage,
        title={'text': "10-Year Metabolic Risk", 'font': {'color': '#ffffff'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 50], 'tickfont': {'color': '#ffffff'}},
            'bar': {'color': "#e74c3c"},
            'steps': [
                {'range': [0, 15], 'color': "#0d3b2e"},
                {'range': [15, 35], 'color': "#3d2e0d"},
                {'range': [35, 50], 'color': "#3d0d0d"}
            ],
            'threshold': {
                'line': {'color': "#f39c12", 'width': 4},
                'thickness': 0.75,
                'value': risk_percentage
            }
        }
    ))
    fig.update_layout(
        height=300,
        paper_bgcolor='#0a1628',
        font={'color': '#ffffff'}
    )
    return fig


def generate_clinical_report(inputs, prediction, metrics, model_name="model", rmr_note=None, scale_note=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""
Clinical Metabolic Health Report
Generated: {now}
Model: {model_name}
---
Patient Summary
---
Predicted RMR: {prediction:.0f} kcal/day
Metabolic Age: {metrics['Metabolic_Age']} years (Actual: {metrics['Actual_Age']} years)
Metabolic Health Score: {metrics['Metabolic_Health_Score']:.0f}/100 ({metrics['Status']})
10-Year Metabolic Risk: {metrics['Risk_Category']} ({metrics['Risk_Percentage']:.0f}%)
BMI: {metrics['BMI']:.1f} kg/m2
Waist-to-Height Ratio: {metrics['Waist_to_Height_Ratio']:.2f}
---
Metabolic Health Components
---
Systolic BP: {metrics['SBP']:.0f} mmHg
Diastolic BP: {metrics['DBP']:.0f} mmHg
Fasting Glucose: {metrics['Glucose']:.0f} mg/dL
HDL Cholesterol: {metrics['HDL']:.0f} mg/dL
Triglycerides: {metrics['Triglycerides']:.0f} mg/dL
---
Exercise Prescription
---
Focus: {metrics['Exercise']['Focus']}
Exercise Type: {metrics['Exercise']['Type']}
Intensity: {metrics['Exercise']['Intensity']}
Weekly Minutes: {metrics['Exercise']['Minutes_Week']} min
Resistance Training: {metrics['Exercise']['Resistance']}
Special Notes: {metrics['Exercise']['Notes']}
---
Recommendations
---
1. Follow the exercise prescription above
2. Monitor caloric intake based on your RMR
3. Regular metabolic health screening every 6-12 months
4. Track changes in weight, blood pressure, and glucose
5. Consult a healthcare provider for personalized advice
---
This report is for educational and research purposes only.
Not a substitute for professional medical advice.
"""
    if scale_note:
        report += f"\nScale Correction Note\n---\n{scale_note}\n"
    if rmr_note:
        report += f"\nData Quality Note\n---\n{rmr_note}\n"
    return report


# ============================================================
# MAIN
# ============================================================
def main():
    render_header()
    model, model_name = load_model()

    with st.sidebar:
        st.markdown("## Metabolic Health")
        st.markdown("---")
        st.markdown(f"**Model:** {model_name}")
        st.markdown("**Data:** NHANES 2017-2018")
        st.markdown("**Sample:** 8,704 participants")
        st.markdown("---")
        st.markdown("### Features")
        st.markdown("- RMR Prediction")
        st.markdown("- Metabolic Health Score")
        st.markdown("- Metabolic Age")
        st.markdown("- Exercise Prescription")
        st.markdown("- 10-Year Risk Assessment")
        st.markdown("- Clinical Report")
        st.markdown("---")
        show_debug = st.checkbox("Show model diagnostics", value=False)
        st.caption("Version 2.0 | For Research and Educational Use")

    if model is None:
        st.warning(
            "No trained model found at `models/best_model.pkl`. "
            "Run the full pipeline first: `python run_pipeline.py`"
        )
        st.stop()

    st.success(f"Loaded best model: {model_name}")

    inputs = patient_input_form()
    if inputs is None:
        st.info("Enter patient information and click 'Generate Comprehensive Metabolic Report'")
        return

    try:
        inputs_aligned = align_features(inputs, model)
        raw_prediction = float(model.predict(inputs_aligned)[0])
        metrics = compute_advanced_metrics(inputs)

        # Step 1: Auto-detect scaling bug
        scaled_prediction, scale_factor, scale_note = detect_and_correct_scale(
            raw_prediction, metrics["RMR_Estimated"], model_name
        )

        # Step 2: Final sanity guard
        prediction, rmr_note, used_fallback = get_safe_rmr(scaled_prediction, metrics["RMR_Estimated"])

        if scale_note and not used_fallback:
            st.info(scale_note)

        if used_fallback:
            st.error(
                "The model's RMR prediction looked physiologically implausible "
                f"even after checking for scaling issues (raw: {raw_prediction:.0f} kcal/day), "
                "so the report below is showing the formula-based estimate instead. "
                "Enable 'Show model diagnostics' in the sidebar for details."
            )

        if show_debug:
            with st.expander("Model diagnostics", expanded=True):
                st.write("Raw model prediction (kcal/day):", raw_prediction)
                st.write("Scale factor applied:", scale_factor)
                st.write("Scale-corrected prediction (kcal/day):", scaled_prediction)
                st.write("Formula-based estimate (kcal/day):", metrics["RMR_Estimated"])
                st.write("Final displayed value (kcal/day):", prediction)
                st.write("Used formula fallback:", used_fallback)
                st.write("Features sent to model:")
                st.dataframe(inputs_aligned)

        st.markdown("---")
        st.subheader("Comprehensive Metabolic Health Report")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{prediction:.0f}</div>
                    <div class="metric-label">Predicted RMR</div>
                    <div class="metric-sub">kcal/day</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            status_class = "badge-good" if metrics["Metabolic_Health_Score"] >= 70 else "badge-fair" if metrics["Metabolic_Health_Score"] >= 50 else "badge-poor"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics["Metabolic_Health_Score"]:.0f}/100</div>
                    <div class="metric-label">Metabolic Health</div>
                    <div class="metric-sub"><span class="{status_class}">{metrics["Status"]}</span></div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            age_diff = metrics['Actual_Age'] - metrics['Metabolic_Age']
            age_text = f"{'Younger' if age_diff > 0 else 'Older'}" if age_diff != 0 else "Same"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics["Metabolic_Age"]} yrs</div>
                    <div class="metric-label">Metabolic Age</div>
                    <div class="metric-sub">Actual: {metrics["Actual_Age"]} yrs ({age_text})</div>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            risk_class = "badge-low" if metrics["Risk_Category"] == "Low" else "badge-moderate" if metrics["Risk_Category"] == "Moderate" else "badge-high"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics["Risk_Percentage"]:.0f}%</div>
                    <div class="metric-label">10-Year Risk</div>
                    <div class="metric-sub"><span class="{risk_class}">{metrics["Risk_Category"]}</span></div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "RMR Analysis",
            "Metabolic Health",
            "Exercise Rx",
            "Risk Assessment",
            "Clinical Report"
        ])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### RMR Comparison")
                fig = create_rmr_comparison_plot(prediction)
                st.pyplot(fig)

                st.markdown("### RMR Interpretation")
                if prediction > 1500:
                    st.success(f"Your RMR ({prediction:.0f} kcal/day) is above population average.")
                elif prediction < 1400:
                    st.warning(f"Your RMR ({prediction:.0f} kcal/day) is below population average.")
                else:
                    st.info(f"Your RMR ({prediction:.0f} kcal/day) is at population average.")

            with col2:
                st.markdown("### Key Factors")
                st.markdown("""
                | Factor | Impact |
                |--------|--------|
                | Weight | Strongest predictor |
                | Age | Moderate predictor |
                | Sex | Moderate predictor |
                | Height | Moderate predictor |
                """)

                st.markdown("### RMR Details")
                source_label = "Formula-based estimate (model output was flagged as implausible)" if used_fallback else (
                    f"Trained model, scale-corrected by {scale_factor:g}x" if scale_factor != 1.0 else "Trained model"
                )
                st.markdown(f"""
                - **Predicted RMR:** {prediction:.0f} kcal/day
                - **Mifflin-St Jeor Equation:** Used for target
                - **Source:** {source_label}
                """)

                age_gap = metrics['Actual_Age'] - metrics['Metabolic_Age']
                if age_gap > 0:
                    st.success(f"Your metabolic age is {age_gap} years younger than your actual age.")
                elif age_gap < 0:
                    st.warning(f"Your metabolic age is {abs(age_gap)} years older than your actual age.")
                else:
                    st.info("Your metabolic age matches your actual age.")

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Metabolic Health Score")
                fig = create_metabolic_health_gauge(metrics["Metabolic_Health_Score"])
                st.plotly_chart(fig, use_container_width=True)

                if metrics["Metabolic_Health_Score"] >= 70:
                    st.success("Good metabolic health - Continue maintaining healthy lifestyle")
                elif metrics["Metabolic_Health_Score"] >= 50:
                    st.warning("Moderate metabolic health - Consider lifestyle improvements")
                else:
                    st.error("Needs improvement - Consult healthcare provider for guidance")

            with col2:
                st.markdown("### Component Analysis")
                fig = create_component_radar(metrics)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Component Breakdown")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("BMI", f"{metrics['BMI']:.1f} kg/m2",
                         delta="Normal" if metrics['BMI'] < 25 else "High")
                st.metric("Waist-to-Height", f"{metrics['Waist_to_Height_Ratio']:.2f}",
                         delta="Good" if metrics['Waist_to_Height_Ratio'] < 0.5 else "High")
            with col2:
                st.metric("Systolic BP", f"{metrics['SBP']:.0f} mmHg",
                         delta="Normal" if metrics['SBP'] < 120 else "Elevated")
                st.metric("Diastolic BP", f"{metrics['DBP']:.0f} mmHg",
                         delta="Normal" if metrics['DBP'] < 80 else "Elevated")
            with col3:
                st.metric("Fasting Glucose", f"{metrics['Glucose']:.0f} mg/dL",
                         delta="Normal" if metrics['Glucose'] < 100 else "Elevated")
                st.metric("HDL Cholesterol", f"{metrics['HDL']:.0f} mg/dL",
                         delta="Good" if metrics['HDL'] > 40 else "Low")

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Exercise Prescription")
                ex = metrics["Exercise"]

                st.markdown(f"""
                | Parameter | Recommendation |
                |-----------|----------------|
                | **Focus** | {ex['Focus']} |
                | **Exercise Type** | {ex['Type']} |
                | **Intensity** | {ex['Intensity']} |
                | **Weekly Minutes** | {ex['Minutes_Week']} min |
                | **Resistance Training** | {ex['Resistance']} |
                """)

                st.info(f"Recommendation: {ex['Notes']}")

            with col2:
                st.markdown("### Weekly Activity Plan")
                fig = create_weekly_schedule(ex)
                st.pyplot(fig)

            st.markdown("### Exercise Guidelines")
            st.markdown("""
            #### Cardio Recommendations
            - Moderate intensity: 30-45 min, 5x/week
            - Vigorous intensity: 20-30 min, 3x/week
            - Warm-up: 5-10 min before each session
            - Cool-down: 5-10 min after each session

            #### Resistance Training
            - Frequency: 2-3x/week
            - Exercises: Compound movements (squats, push-ups, rows)
            - Sets: 3-4 sets of 8-12 reps
            - Rest: 60-90 seconds between sets
            """)

        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 10-Year Metabolic Risk")
                fig = create_risk_gauge(metrics["Risk_Percentage"])
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Risk Breakdown")

                risk_score = 0
                if metrics['BMI'] > 30:
                    risk_score += 2
                elif metrics['BMI'] > 25:
                    risk_score += 1
                if metrics['Metabolic_Health_Score'] < 50:
                    risk_score += 2
                elif metrics['Metabolic_Health_Score'] < 70:
                    risk_score += 1
                if metrics['Actual_Age'] > 50:
                    risk_score += 1
                if metrics['Actual_Age'] > 65:
                    risk_score += 2
                if prediction < 1400:
                    risk_score += 1
                if prediction < 1200:
                    risk_score += 2

                st.metric("Risk Score", f"{risk_score}/10")

                risk_factors = []
                if metrics['BMI'] > 30:
                    risk_factors.append("BMI > 30 (Obesity)")
                elif metrics['BMI'] > 25:
                    risk_factors.append("BMI 25-30 (Overweight)")
                if metrics['Metabolic_Health_Score'] < 50:
                    risk_factors.append("Poor Metabolic Health")
                elif metrics['Metabolic_Health_Score'] < 70:
                    risk_factors.append("Fair Metabolic Health")
                if metrics['Actual_Age'] > 50:
                    risk_factors.append("Age > 50 years")
                if metrics['Actual_Age'] > 65:
                    risk_factors.append("Age > 65 years")
                if prediction < 1400:
                    risk_factors.append("Low RMR (< 1400 kcal/day)")

                if risk_factors:
                    st.markdown("**Risk Factors Identified:**")
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
                else:
                    st.success("No major risk factors identified.")

            st.markdown("### Risk Interpretation")
            if metrics["Risk_Category"] == "Low":
                st.success("""
                Low Risk (10-Year Metabolic Decline <10%)

                Your metabolic health is good. Continue maintaining:
                - Regular physical activity
                - Healthy diet
                - Regular health screenings
                """)
            elif metrics["Risk_Category"] == "Moderate":
                st.warning("""
                Moderate Risk (10-Year Metabolic Decline 10-25%)

                Consider lifestyle improvements:
                - Increase physical activity
                - Monitor diet and weight
                - Regular metabolic health checks
                - Consult healthcare provider
                """)
            else:
                st.error("""
                High Risk (10-Year Metabolic Decline >25%)

                Immediate action recommended:
                - Consult healthcare provider
                - Comprehensive metabolic evaluation
                - Structured exercise program
                - Dietary intervention
                - Regular monitoring
                """)

        with tab5:
            st.markdown("### Clinical Report")

            report = generate_clinical_report(inputs, prediction, metrics, model_name, rmr_note, scale_note if not used_fallback else None)

            with st.expander("Preview Report", expanded=True):
                st.text(report)

            st.download_button(
                label="Download Clinical Report (Markdown)",
                data=report,
                file_name=f"metabolic_health_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True
            )

            st.download_button(
                label="Download Report (TXT)",
                data=report,
                file_name=f"metabolic_health_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )

            st.info("The report includes: RMR analysis, metabolic health score, exercise prescription, and 10-year risk assessment.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Please check your inputs and try again.")


if __name__ == "__main__":
    main()