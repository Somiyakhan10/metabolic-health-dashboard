"""
explain.py
==========
Explainable AI suite: SHAP (global + local + interaction), permutation
importance, and partial dependence plots (1D + 2D interaction).
"""

from __future__ import annotations

import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Physiological rationale library used to auto-annotate top features.
CLINICAL_RATIONALE = {
    "BMXWT": "Body weight is the dominant determinant of RMR because it is a proxy for total metabolically active tissue (mostly fat-free mass).",
    "BMXHT": "Height contributes to RMR via its relationship with fat-free mass and body surface area.",
    "RIDAGEYR": "RMR declines with age due to progressive loss of fat-free mass (sarcopenia) and reduced organ metabolic activity.",
    "RIAGENDR": "Men typically have higher RMR than women at the same weight/height due to greater average fat-free mass.",
    "BMXBMI": "BMI reflects overall adiposity; higher BMI is generally associated with higher absolute RMR (more tissue to maintain) but lower RMR per kg.",
    "BMXWAIST": "Waist circumference indexes visceral adiposity, which carries its own metabolic activity distinct from BMI.",
    "Waist_to_Height_Ratio": "WHtR captures central adiposity independent of overall body size, a stronger cardiometabolic risk marker than BMI alone.",
    "LBXGLU": "Fasting glucose reflects glycemic status; chronic hyperglycemia is linked to altered substrate utilization and metabolic rate.",
    "LBXGH": "HbA1c reflects long-term glycemic control and is associated with metabolic dysregulation that can influence energy expenditure.",
    "HOMA_IR": "Insulin resistance (HOMA-IR) is linked to altered substrate oxidation and can modestly influence resting energy expenditure.",
    "LBDHDD": "HDL cholesterol is a marker of lipid metabolism and cardiometabolic health, indirectly associated with metabolic rate.",
    "LBXTR": "Triglycerides reflect lipid metabolism status, part of the broader metabolic syndrome cluster.",
    "BPXSY1": "Systolic blood pressure is a component of metabolic syndrome and correlates with overall metabolic burden.",
    "Metabolic_Syndrome_Flag": "Presence of metabolic syndrome indicates clustered cardiometabolic dysfunction that may alter energy metabolism.",
    "BMI_x_Age": "Interaction term capturing how the adiposity-metabolism relationship shifts across the lifespan.",
    "Glucose_x_HbA1c": "Interaction capturing combined acute and chronic glycemic burden on metabolic processes.",
}


def compute_shap_values(model, X_background: pd.DataFrame, X_explain: pd.DataFrame):
    """Compute SHAP values using the most appropriate explainer.

    Uses TreeExplainer for tree-based models (fast, exact) and falls back
    to KernelExplainer (model-agnostic, slower) otherwise.

    Args:
        model: Fitted regressor.
        X_background: Background/reference sample for the explainer.
        X_explain: Rows to compute SHAP values for.

    Returns:
        shap.Explanation object.
    """
    model_type = type(model).__name__.lower()
    if any(k in model_type for k in ["forest", "xgb", "lgbm", "catboost", "gbm", "boost"]):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_background, 100))
    return explainer(X_explain)


def plot_shap_summary(shap_values, X_explain: pd.DataFrame, out_path: str) -> str:
    """Save a SHAP global feature-importance summary (beeswarm) plot.

    Args:
        shap_values: shap.Explanation from compute_shap_values.
        X_explain: Feature matrix corresponding to shap_values.
        out_path: PNG output path.

    Returns:
        The out_path (for chaining/reporting).
    """
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_explain, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_shap_waterfall(shap_values, index: int, out_path: str) -> str:
    """Save a SHAP waterfall plot explaining a single prediction.

    Args:
        shap_values: shap.Explanation object.
        index: Row index within shap_values to explain.
        out_path: PNG output path.

    Returns:
        The out_path.
    """
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_values[index], show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_shap_dependence(shap_values, X_explain: pd.DataFrame, feature: str, out_path: str) -> str:
    """Save a SHAP dependence plot for a single feature.

    Args:
        shap_values: shap.Explanation object.
        X_explain: Feature matrix.
        feature: Feature name to plot.
        out_path: PNG output path.

    Returns:
        The out_path.
    """
    plt.figure(figsize=(7, 5))
    shap.dependence_plot(feature, shap_values.values, X_explain, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def compute_permutation_importance(model, X: pd.DataFrame, y: pd.Series, config: dict) -> pd.DataFrame:
    """Compute permutation feature importance with standard deviations.

    Args:
        model: Fitted estimator.
        X: Feature matrix (held-out set recommended).
        y: True target values.
        config: Project configuration (uses random_seed).

    Returns:
        DataFrame with columns [feature, importance_mean, importance_std],
        sorted descending by importance.
    """
    result = permutation_importance(
        model, X, y, n_repeats=20, random_state=config["random_seed"], scoring="r2", n_jobs=-1
    )
    df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


def plot_partial_dependence(model, X: pd.DataFrame, features: List[str], out_path: str) -> str:
    """Save 1D partial dependence plots for the top features.

    Args:
        model: Fitted estimator.
        X: Feature matrix used to compute PDPs.
        features: List of feature names (top-N) to plot.
        out_path: PNG output path.

    Returns:
        The out_path.
    """
    fig, ax = plt.subplots(figsize=(4 * len(features), 4), dpi=300)
    PartialDependenceDisplay.from_estimator(model, X, features, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_2d_interaction_pdp(model, X: pd.DataFrame, feature_pair: tuple, out_path: str) -> str:
    """Save a 2D interaction partial dependence plot for a feature pair.

    Args:
        model: Fitted estimator.
        X: Feature matrix.
        feature_pair: Tuple of two feature names, e.g. ("RIDAGEYR", "BMXBMI").
        out_path: PNG output path.

    Returns:
        The out_path.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    PartialDependenceDisplay.from_estimator(model, X, [feature_pair], ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def clinical_interpretation(top_features: List[str]) -> pd.DataFrame:
    """Attach physiological rationale text to a list of top features.

    Args:
        top_features: Ordered list of important feature names.

    Returns:
        DataFrame with [feature, rationale] columns; unmapped features get
        a generic placeholder rationale.
    """
    rows = [
        {"feature": f, "rationale": CLINICAL_RATIONALE.get(
            f, "No pre-authored rationale on file — interpret in context of known metabolic physiology.")}
        for f in top_features
    ]
    return pd.DataFrame(rows)
