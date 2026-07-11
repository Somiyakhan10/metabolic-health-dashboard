"""
train_models.py
================
Trains and compares Linear Regression, Elastic Net, Random Forest, XGBoost,
LightGBM, and CatBoost for RMR regression, with 5-fold CV grid search and
automatic best-model selection by mean CV R^2.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, KFold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


def build_model_grid(config: dict) -> Dict[str, dict]:
    """Assemble estimator + hyperparameter-grid pairs from config.

    Args:
        config: Project configuration (uses the ``models`` section).

    Returns:
        Dict mapping model name -> {"estimator": ..., "param_grid": ...,
        "uses_scaled": "standard"|"minmax"|"none"}.
    """
    seed = config["random_seed"]
    m = config["models"]
    grid: Dict[str, dict] = {
        "linear_regression": {
            "estimator": LinearRegression(),
            "param_grid": {},
            "uses_scaled": "minmax",
        },
        "elastic_net": {
            "estimator": ElasticNet(random_state=seed, max_iter=10000),
            "param_grid": {"alpha": m["elastic_net"]["alpha"], "l1_ratio": m["elastic_net"]["l1_ratio"]},
            "uses_scaled": "minmax",
        },
        "random_forest": {
            "estimator": RandomForestRegressor(random_state=seed, n_jobs=-1),
            "param_grid": {"n_estimators": m["random_forest"]["n_estimators"],
                            "max_depth": m["random_forest"]["max_depth"]},
            "uses_scaled": "none",
        },
    }

    if XGBRegressor is not None:
        grid["xgboost"] = {
            "estimator": XGBRegressor(random_state=seed, n_jobs=-1, objective="reg:squarederror"),
            "param_grid": {"n_estimators": m["xgboost"]["n_estimators"],
                            "max_depth": m["xgboost"]["max_depth"],
                            "learning_rate": m["xgboost"]["learning_rate"]},
            "uses_scaled": "none",
        }
    if LGBMRegressor is not None:
        grid["lightgbm"] = {
            "estimator": LGBMRegressor(random_state=seed, n_jobs=-1),
            "param_grid": {"n_estimators": m["lightgbm"]["n_estimators"],
                            "num_leaves": m["lightgbm"]["num_leaves"],
                            "learning_rate": m["lightgbm"]["learning_rate"]},
            "uses_scaled": "none",
        }
    if CatBoostRegressor is not None:
        grid["catboost"] = {
            "estimator": CatBoostRegressor(random_state=seed, verbose=0),
            "param_grid": {"iterations": m["catboost"]["iterations"],
                            "depth": m["catboost"]["depth"],
                            "learning_rate": m["catboost"]["learning_rate"]},
            "uses_scaled": "none",
        }

    missing = [n for n, v in [("xgboost", XGBRegressor), ("lightgbm", LGBMRegressor),
                               ("catboost", CatBoostRegressor)] if v is None]
    if missing:
        logger.warning("Skipping unavailable libraries (pip install to enable): %s", missing)

    return grid


def train_all_models(splits: dict, config: dict) -> Tuple[Dict[str, GridSearchCV], pd.DataFrame]:
    """Grid-search every configured model with 5-fold CV.

    Args:
        splits: Output dict from preprocess.run_full_preprocessing (must
            contain X_train_standard/minmax and y_train).
        config: Project configuration.

    Returns:
        Tuple of (fitted_models dict keyed by model name, leaderboard
        DataFrame sorted by mean CV R^2 descending).
    """
    cv_cfg = config["cv"]
    kfold = KFold(n_splits=cv_cfg["n_splits"], shuffle=cv_cfg["shuffle"], random_state=config["random_seed"])

    model_grid = build_model_grid(config)
    y_train = splits["y_train"]

    fitted_models: Dict[str, GridSearchCV] = {}
    leaderboard_rows = []

    for name, spec in model_grid.items():
        scale_key = spec["uses_scaled"]
        X_train = splits["X_train_minmax"] if scale_key == "minmax" else splits["X_train_standard"] \
            if scale_key == "standard" else splits["X_train"]

        logger.info("Training %s ...", name)
        search = GridSearchCV(
            estimator=spec["estimator"],
            param_grid=spec["param_grid"] or {"__dummy__": [None]} if not spec["param_grid"] else spec["param_grid"],
            scoring="r2",
            cv=kfold,
            n_jobs=-1,
            refit=True,
        )
        # GridSearchCV requires a non-empty grid; handle the no-hyperparam case
        if not spec["param_grid"]:
            search = spec["estimator"].fit(X_train, y_train)
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(spec["estimator"], X_train, y_train, cv=kfold, scoring="r2")
            mean_r2, std_r2 = scores.mean(), scores.std()
            best_params = {}
            fitted_models[name] = search
        else:
            search.fit(X_train, y_train)
            mean_r2 = search.best_score_
            std_r2 = search.cv_results_["std_test_score"][search.best_index_]
            best_params = search.best_params_
            fitted_models[name] = search.best_estimator_

        leaderboard_rows.append({
            "model": name, "mean_cv_r2": mean_r2, "std_cv_r2": std_r2, "best_params": best_params
        })
        logger.info("%s -> CV R^2 = %.4f (+/- %.4f)", name, mean_r2, std_r2)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("mean_cv_r2", ascending=False).reset_index(drop=True)
    return fitted_models, leaderboard


def select_and_save_best_model(fitted_models: dict, leaderboard: pd.DataFrame, config: dict) -> str:
    """Persist the top-ranked model from the leaderboard to disk.

    Args:
        fitted_models: Dict of fitted estimators keyed by model name.
        leaderboard: Ranked comparison DataFrame from train_all_models.
        config: Project configuration.

    Returns:
        Path to the saved .pkl file.
    """
    best_name = leaderboard.iloc[0]["model"]
    best_model = fitted_models[best_name]
    out_path = f"{config['paths']['model_dir']}/best_model.pkl"
    joblib.dump({"model": best_model, "name": best_name}, out_path)
    logger.info("Best model: %s (CV R^2=%.4f) saved to %s",
                best_name, leaderboard.iloc[0]["mean_cv_r2"], out_path)
    return out_path
