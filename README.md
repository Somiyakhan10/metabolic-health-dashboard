# Complete README.md File

Copy this entire file and paste it directly into your `README.md`:

```markdown
# Explainable AI for Personalized Resting Metabolic Rate Prediction

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Data](https://img.shields.io/badge/data-NHANES%202017--2018-orange)

An explainable machine learning pipeline for predicting Resting Metabolic Rate (RMR) from demographic, anthropometric, laboratory, and behavioral data in the NHANES 2017-2018 cohort.

---

## Dashboard Preview

| Main Dashboard | RMR Analysis |
|----------------|--------------|
| <img width="1867" height="789" alt="image" src="https://github.com/user-attachments/assets/cdf9fc92-d065-4796-b83a-424affc4ce74" />
 | ![RMR Analysis](images/dashboard_rmr.png) |

| Metabolic Health | Exercise Prescription |
|------------------|----------------------|
| <img width="1885" height="817" alt="image" src="https://github.com/user-attachments/assets/6364fdce-ff59-4686-af6c-cf579c5f5ff7" />
 | ![Exercise Prescription](images/dashboard_exercise.png) |

| Risk Assessment | Clinical Report |
|-----------------|-----------------|
| <img width="1377" height="580" alt="image" src="https://github.com/user-attachments/assets/4cdd98c2-5fe9-49b8-8bf9-063e63b66548" />
 | ![Clinical Report](images/dashboard_report.png) |

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
git clone <your-repo-url>
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

## Project Structure

```
rmr_prediction_project/
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── src/
│   ├── data_loader.py      # NHANES data loading
│   ├── feature_engineering.py
│   ├── preprocess.py
│   ├── train_models.py
│   ├── evaluate.py
│   └── explain.py          # SHAP explainability
├── models/
│   └── best_model.pkl
├── data/
│   ├── raw/                # NHANES .XPT files
│   └── processed/
├── images/                 # Dashboard screenshots
│   ├── dashboard_main.png
│   ├── dashboard_rmr.png
│   ├── dashboard_health.png
│   ├── dashboard_exercise.png
│   ├── dashboard_risk.png
│   └── dashboard_report.png
├── outputs/
│   └── figures/            # Generated plots
├── report/
│   └── manuscript.md       # Academic manuscript
├── requirements.txt
├── config.yaml
├── run_pipeline.py
└── README.md
```

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


Somiya Khan. (2026). Explainable AI for Personalized Resting Metabolic 
Rate Prediction Using NHANES 2017-2018 Data. GitHub repository.
```


