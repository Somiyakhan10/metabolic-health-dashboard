"""
evaluate.py
===========
Regression metrics, stratified cross-validation, and learning curves.
"""

from __future__ import annotations

import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
    """Compute adjusted R^2, penalizing for the number of predictors.

    Args:
        r2: Standard R^2.
        n_samples: Number of observations.
        n_features: Number of predictor features.

    Returns:
        Adjusted R^2.
    """
    if n_samples - n_features - 1 <= 0:
        return np.nan
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> Dict[str, float]:
    """Compute the full regression metric suite.

    Args:
        y_true: Ground-truth RMR values.
        y_pred: Model predictions.
        n_features: Number of features used (for adjusted R^2).

    Returns:
        Dict with mae, rmse, r2, mape, adj_r2.
    """
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2,
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "Adjusted_R2": adjusted_r2(r2, len(y_true), n_features),
    }


def evaluate_all_models(fitted_models: dict, splits: dict, config: dict) -> pd.DataFrame:
    """Evaluate every fitted model on the held-out test set.

    Args:
        fitted_models: Dict of model_name -> fitted estimator.
        splits: Preprocessing output (must contain X_test/_standard/_minmax, y_test).
        config: Project configuration.

    Returns:
        DataFrame of test-set metrics per model.
    """
    rows = []
    for name, model in fitted_models.items():
        if name in ("linear_regression", "elastic_net"):
            X_test = splits["X_test_minmax"]
        else:
            # Tree-based models (random_forest, xgboost, lightgbm, catboost)
            # are trained on unscaled features in train_models.py.
            X_test = splits["X_test"]
        y_pred = model.predict(X_test)
        metrics = compute_regression_metrics(splits["y_test"].values, y_pred, X_test.shape[1])
        metrics["model"] = name
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("model").sort_values("R2", ascending=False)


def stratified_kfold_by_bmi(X: pd.DataFrame, bmi_category: pd.Series, config: dict):
    """Yield stratified K-fold splits, stratifying on BMI category.

    Args:
        X: Feature matrix (index-aligned with bmi_category).
        bmi_category: Categorical BMI-category labels for stratification.
        config: Project configuration (uses cv.n_splits).

    Yields:
        (train_idx, test_idx) tuples.
    """
    skf = StratifiedKFold(n_splits=config["cv"]["n_splits"], shuffle=True, random_state=config["random_seed"])
    yield from skf.split(X, bmi_category)


def plot_learning_curve(estimator, X: pd.DataFrame, y: pd.Series, model_name: str, out_dir: str) -> str:
    """Plot and save a training-vs-validation learning curve.

    Args:
        estimator: A fitted or unfitted sklearn-compatible regressor.
        X: Full feature matrix.
        y: Target vector.
        model_name: Name used in title and filename.
        out_dir: Directory to save the PNG figure.

    Returns:
        Path to the saved figure.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training R2")
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation R2")
    ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15)
    ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("R^2 score")
    ax.set_title(f"Learning Curve — {model_name}")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path = f"{out_dir}/learning_curve_{model_name}.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved learning curve to %s", out_path)
    return out_path
