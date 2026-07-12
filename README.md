# Explainable AI for Personalized Resting Metabolic Rate Prediction

An explainable machine learning pipeline for predicting Resting Metabolic Rate (RMR) from demographic, anthropometric, laboratory, and behavioral data in the NHANES 2017-2018 cohort.

<p align="center">
  <a href="https://metabolic-health-dashboard.streamlit.app/">
    <img src="https://img.shields.io/badge/LIVE_DEMO-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
</p>


## Dashboard Preview

| Main Dashboard | RMR Analysis |
|----------------|--------------|
| <img width="1867" height="789" alt="image" src="https://github.com/user-attachments/assets/cdf9fc92-d065-4796-b83a-424affc4ce74" />
 

| Metabolic Health | Exercise Prescription |
|------------------|----------------------|
| <img width="1331" height="671" alt="image" src="https://github.com/user-attachments/assets/48438407-50e8-40e1-a268-ea5e6ac5543d" />

 

| Risk Assessment | Clinical Report |
|-----------------|-----------------|
| <img width="984" height="549" alt="image" src="https://github.com/user-attachments/assets/5f65c266-eed9-4cd5-a8c1-235df1bbcfd6" />


---

## Quick Overview

| Feature | Description |
|---------|-------------|
| **RMR Prediction** | Predicts Resting Metabolic Rate using ML models |
| **Metabolic Health Score** | 0-100 score based on 5 clinical components |
| **Metabolic Age** | Compares metabolic age to chronological age |
| **10-Year Risk** | Risk assessment for metabolic syndrome |
| **Exercise Prescription** | Personalized exercise recommendations |
| **SHAP Explainability** | Understand why predictions are made |
| **Clinical Report** | Downloadable healthcare reports |

---

## Model Performance

| Model | CV R² |
|-------|-------|
| XGBoost | 0.9866 |
| Random Forest | 0.9706 |
| CatBoost | 0.96+ |
| Linear Regression | 0.90 |

---

## Installation

```bash
# Clone repository
git clone <https://github.com/Somiyakhan10/metabolic-health-dashboard>
cd rmr_prediction_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Run the complete pipeline
python run_pipeline.py

# Launch the dashboard
streamlit run dashboard/app.py
```

Open your browser to `http://localhost:8501`

---

## Data Source

NHANES 2017-2018 (8,704 participants)
- Demographics, Body Measurements, Blood Pressure
- Fasting Glucose, HbA1c, Lipids
- Physical Activity, Diabetes Status

---

## Key Features

- Multiple ML Models: XGBoost, Random Forest, LightGBM, CatBoost
- SHAP Explainability: Global and local explanations
- Physiological Plausibility Checks: Clinically safe predictions
- Interactive Dashboard: Professional dark theme
- Clinical Report Generation: Downloadable reports

---

## Limitations

- RMR target is equation-derived (Mifflin-St Jeor), not directly measured
- Cross-sectional NHANES data (no causal claims)
- Self-reported physical activity
- Limited to U.S. NHANES 2017-2018 sampling frame

---

## Future Work

- Validate against directly measured RMR (indirect calorimetry)
- Incorporate additional NHANES cycles
- Add body composition biomarkers
- Deploy as clinical API



