# Explainable AI for Personalized Resting Metabolic Rate Prediction Using NHANES 2017–2018 Data

**Author:** [Your Name]
**Affiliation:** [Your Institution]

---

## Abstract

*(Structured, ~250 words. Fill in bracketed values after running the pipeline on real data.)*

**Background:** Resting metabolic rate (RMR) is a cornerstone measurement
in obesity, metabolic disease, and sports nutrition, yet direct
measurement via indirect calorimetry is impractical at scale. Predictive
equations such as Mifflin-St Jeor remain the clinical default despite
using only four inputs (weight, height, age, sex). **Objective:** To
develop and interpret a machine-learning framework for RMR estimation
using a nationally representative U.S. sample, testing whether inclusion
of metabolic and behavioral covariates yields additional, interpretable
signal beyond the standard predictive equation. **Methods:** We merged
ten NHANES 2017–2018 components (demographics, body measurements, blood
pressure, fasting glucose, HbA1c, HDL, total cholesterol, triglycerides
and insulin, physical activity, and diabetes history) on participant ID
(SEQN), yielding an analytic sample of N=[XXXX] after preprocessing.
RMR was computed via the Mifflin-St Jeor equation as the modeling target.
Six regression approaches (Linear Regression, Elastic Net, Random Forest,
XGBoost, LightGBM, CatBoost) were compared via 5-fold cross-validation;
the best model was evaluated on a held-out test set (MAE, RMSE, R²,
MAPE, adjusted R²) and interpreted using SHAP values, permutation
importance, and partial dependence plots. **Results:** The best-performing
model was [MODEL NAME] (test R² = [X.XX], MAE = [XX] kcal/day). SHAP
analysis confirmed weight, height, age, and sex as dominant predictors,
consistent with the Mifflin-St Jeor formula's structure, with
[secondary feature, e.g., waist circumference / HOMA-IR] contributing
meaningful additional explanatory signal. **Conclusion:** Explainable
ML applied to richly phenotyped population data can recover expected
physiological determinants of RMR while quantifying the marginal
contribution of metabolic covariates, offering a transparent framework
for individualized metabolic risk communication pending validation
against directly measured RMR.

---

## 1. Introduction

Resting metabolic rate (RMR), the energy expended to sustain basic
physiological processes at rest, accounts for 60–75% of total daily
energy expenditure in most adults and is a foundational quantity in
weight management, clinical nutrition therapy, and metabolic disease
research (Frankenfield et al., 2005). Because indirect calorimetry — the
gold-standard RMR measurement — requires specialized equipment and
trained staff, clinicians and researchers routinely substitute predictive
equations. The Mifflin-St Jeor equation (Mifflin et al., 1990) is widely
regarded as the most accurate of the classical equations across body
weight categories (Frankenfield et al., 2005) and uses only four inputs:
weight, height, age, and sex.

This simplicity is also a limitation: it cannot capture known
metabolic heterogeneity — e.g., the reduced fat-free mass and altered
substrate metabolism seen in insulin resistance, or central adiposity's
distinct metabolic signature relative to overall BMI (Camhi et al.,
2011). Machine learning offers a route to model such heterogeneity from
observational data, but has historically been criticized as a "black
box" unsuitable for clinical translation. Explainable AI (XAI) methods,
particularly SHapley Additive exPlanations (SHAP; Lundberg & Lee, 2017),
directly address this by attributing each prediction to individual
feature contributions in a way clinicians and researchers can audit.

**Objectives:** (1) Build a merged, feature-rich NHANES 2017–2018
dataset spanning demographic, anthropometric, laboratory, and behavioral
domains; (2) train and compare multiple regression algorithms against a
Mifflin-St Jeor–derived RMR target; (3) apply a comprehensive XAI suite
(SHAP, permutation importance, partial dependence) to test whether the
learned model recovers expected physiology and surfaces interpretable
secondary predictors; (4) package the result as a clinician-facing,
explainable dashboard.

## 2. Related Work

**RMR prediction equations.** The Harris-Benedict (1919), Mifflin-St
Jeor (1990), and Owen (1986, 1987) equations remain in wide clinical
use. Comparative validation studies consistently rank Mifflin-St Jeor as
the most accurate against indirect calorimetry in non-obese and obese
adults alike (Frankenfield et al., 2005), motivating its use here as the
best available equation-based ground truth in the absence of measured
RMR in NHANES.

**Machine learning in metabolic health.** ML approaches have been
applied to predict metabolic syndrome, insulin resistance, and energy
expenditure from NHANES and similar cohorts, generally outperforming
linear approaches on non-linear or interacting risk factors (e.g., HOMA-IR
and central adiposity measures) while raising interpretability concerns
that XAI methods are increasingly used to address (Lundberg et al.,
2020).

**Explainable AI in clinical prediction.** SHAP and permutation
importance have become standard tools for auditing clinical ML models,
enabling both global (population-level) and local (patient-level)
interpretation, and are increasingly required by reviewers and regulators
for clinical decision-support tools (Lundberg et al., 2020).

## 3. Methods

### 3.1 Dataset

NHANES 2017–2018 components DEMO_J, BMX_J, BPX_J, GLU_J, GHB_J, HDL_J,
TCHOL_J, TRIGLY_J, PAQ_J, and DIQ_J were merged on SEQN via inner join
(`src/data_loader.py`), yielding a starting sample of up to N≈9,254
before preprocessing-related exclusions.

### 3.2 Target construction

RMR (kcal/day) was computed via the Mifflin-St Jeor equation
(`src/feature_engineering.py::compute_rmr_mifflin_st_jeor`), using
weight (BMXWT), height (BMXHT), age (RIDAGEYR), and sex (RIAGENDR).
Because the target is itself a deterministic function of four features,
this is stated explicitly as a design choice rather than obscured: model
performance close to ceiling on these four variables is expected and
diagnostic of correct implementation, while SHAP is used to characterize
the incremental role of the remaining ~15+ engineered/laboratory
features.

### 3.3 Feature engineering

`src/feature_engineering.py` derives: HOMA-IR (insulin resistance),
waist-to-height ratio, a binary metabolic syndrome flag (AHA/NHLBI ATP
III criteria: ≥3 of elevated waist circumference, triglycerides, blood
pressure, fasting glucose, or reduced HDL), physical activity category
(Sedentary/Moderate/Vigorous from PAQ650/PAQ665), diabetes status (from
DIQ010), WHO BMI category, and two interaction terms (BMI×Age,
Glucose×HbA1c).

### 3.4 Preprocessing

`src/preprocess.py` implements: dropping columns with >40% missingness;
median imputation (numeric) / mode imputation (categorical); IQR-based
(1.5×IQR) outlier removal; one-hot encoding of categorical features;
removal of one feature from any pair correlated above r=0.90; Variance
Inflation Factor (VIF) computation to flag remaining multicollinearity
(threshold VIF>10); and a 70/15/15 train/validation/test split stratified
by BMI category, with separate StandardScaler (for linear models) and
MinMaxScaler (retained for completeness) fits on the training set only.

### 3.5 Models

Six regressors were compared: Linear Regression (baseline), Elastic Net
(grid search over α ∈ {0.001,...,10} and l1_ratio ∈ {0.1,...,0.9}),
Random Forest (n_estimators ∈ {100,200,300}), and gradient-boosted trees
XGBoost, LightGBM, and CatBoost (each grid-searched over tree count,
depth, and learning rate). Model selection used 5-fold cross-validated
R² on the training set (`src/train_models.py`); the top model by mean
CV R² was refit and evaluated once on the untouched test set.

### 3.6 Evaluation

Test-set metrics: MAE, RMSE, R², MAPE, and adjusted R² (penalizing for
feature count). Learning curves (training vs. validation R² across
sample-size fractions) were plotted to assess bias/variance trade-offs
(`src/evaluate.py`).

### 3.7 Explainability

`src/explain.py` computes: SHAP values (TreeExplainer for tree models),
global summary (beeswarm) plots, per-patient waterfall plots, SHAP
dependence plots (including Age×BMI interaction), permutation importance
with 20 repeats, and 1D/2D partial dependence plots for the top 5
features. Each top feature is paired with a physiological rationale
(see `CLINICAL_RATIONALE` dictionary) grounded in the exercise
physiology/metabolism literature.

## 4. Results

*(To be completed after execution on real NHANES data.)*

- Final analytic sample size after preprocessing: N = [XXXX]
- Descriptive statistics table (age, sex distribution, BMI, RMR): [Table 1]
- Model comparison leaderboard (CV R² and test-set metrics): [Table 2]
- SHAP global importance ranking: [Figure X]
- Representative SHAP waterfall plot(s): [Figure X]
- Partial dependence plots for top 5 features: [Figure X]

## 5. Discussion

*(To be completed after execution.)* Discuss the extent to which the
best model's performance and SHAP attribution match physiological
expectation (weight/height/age/sex dominance), what secondary predictors
(e.g., waist circumference, HOMA-IR, physical activity) contributed
non-trivially, and what this implies for individualized metabolic risk
communication.

## 6. Limitations

1. **Equation-derived target.** RMR is not directly measured in NHANES;
   the target is a deterministic function of four of the model's own
   features, bounding what "prediction accuracy" can mean here and
   requiring independent validation against indirect calorimetry before
   any clinical claim can be made.
2. **Cross-sectional design.** NHANES cannot support causal inference
   about metabolic drivers of RMR.
3. **Self-report bias.** Physical activity (PAQ_J) is self-reported and
   subject to recall and social-desirability bias.
4. **Generalizability.** Results are bounded by the U.S. NHANES
   2017–2018 sampling frame and may not generalize to other populations
   or time periods.
5. **Missing data.** Inner-join merging across 10 components and
   subsequent imputation may introduce selection effects relative to the
   full NHANES sample.

## 7. Future Work

- Validate model predictions against a cohort with directly measured RMR
  via indirect calorimetry.
- Extend to multiple NHANES cycles for larger samples and temporal
  robustness.
- Incorporate body-composition data (e.g., DXA-derived fat-free mass)
  where available, since fat-free mass is the strongest known
  physiological correlate of RMR.
- Prospective validation of the dashboard tool in a clinical or applied
  sports-nutrition setting.

## 8. Conclusion

This project demonstrates a fully reproducible, explainable ML pipeline
for RMR estimation from a richly phenotyped national health survey,
combining rigorous preprocessing, multi-model comparison, and a
comprehensive interpretability suite, packaged for both academic
reporting and clinician-facing use via an interactive dashboard.

## References

*(Representative starter list — expand to 20–30 with your own literature
review; verify all citations independently before submission.)*

1. Mifflin, M. D., St Jeor, S. T., Hill, L. A., Scott, B. J., Daugherty, S. A., & Koh, Y. O. (1990). A new predictive equation for resting energy expenditure in healthy individuals. *American Journal of Clinical Nutrition*, 51(2), 241–247.
2. Harris, J. A., & Benedict, F. G. (1919). *A Biometric Study of Basal Metabolism in Man*. Carnegie Institution of Washington.
3. Frankenfield, D., Roth-Yousey, L., & Compher, C. (2005). Comparison of predictive equations for resting metabolic rate in healthy nonobese and obese adults: a systematic review. *Journal of the American Dietetic Association*, 105(5), 775–789.
4. Owen, O. E., Holup, J. L., D'Alessio, D. A., et al. (1987). A reappraisal of the caloric requirements of men. *American Journal of Clinical Nutrition*, 46(6), 875–885.
5. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
6. Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67.
7. Camhi, S. M., Bray, G. A., Bouchard, C., et al. (2011). The relationship of waist circumference and BMI to visceral, subcutaneous, and total body fat: sex and race differences. *Obesity*, 19(2), 402–408.
8. Grundy, S. M., Cleeman, J. I., Daniels, S. R., et al. (2005). Diagnosis and management of the metabolic syndrome: an American Heart Association/National Heart, Lung, and Blood Institute scientific statement. *Circulation*, 112(17), 2735–2752.
9. Matthews, D. R., Hosker, J. P., Rudenski, A. S., et al. (1985). Homeostasis model assessment: insulin resistance and beta-cell function from fasting plasma glucose and insulin concentrations in man. *Diabetologia*, 28(7), 412–419.
10. Centers for Disease Control and Prevention (CDC). National Health and Nutrition Examination Survey, 2017–2018 Data Documentation, Codebook, and Frequencies. National Center for Health Statistics.
11. *[Add 10–20 further references from your own literature search: additional RMR equation validation studies, ML-in-obesity papers, NHANES methodology papers, physical activity questionnaire validation, XAI methodology papers, etc.]*

---

*Word count target: 2,000–3,000 words (excluding references/tables). This
template currently runs shorter — expand Sections 4–6 substantially once
real results are available, and deepen the literature review in Section 2.*
