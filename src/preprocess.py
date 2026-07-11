"""
preprocess.py
=============
Cleaning, imputation, outlier handling, encoding, scaling, correlation
pruning, multicollinearity (VIF) assessment, and train/val/test splitting.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.40) -> pd.DataFrame:
    """Drop columns whose missing-value fraction exceeds ``threshold``.

    Args:
        df: Input DataFrame.
        threshold: Fraction (0-1) of missingness above which a column is
            dropped.

    Returns:
        DataFrame with high-missingness columns removed.
    """
    missing_frac = df.isna().mean()
    to_drop = missing_frac[missing_frac > threshold].index.tolist()
    if to_drop:
        logger.info("Dropping %d columns with >%.0f%% missing: %s",
                    len(to_drop), threshold * 100, to_drop)
    return df.drop(columns=to_drop)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numeric, mode for categorical.

    Args:
        df: Input DataFrame (post column-dropping).

    Returns:
        Fully imputed DataFrame (no remaining NaNs in retained columns).
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)

    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: List[str], multiplier: float = 1.5) -> pd.DataFrame:
    """Remove rows containing IQR-based outliers in the given columns.

    Args:
        df: Input DataFrame.
        columns: Numeric columns to check for outliers.
        multiplier: IQR multiplier (1.5 = standard Tukey fence).

    Returns:
        DataFrame with outlier rows removed.
    """
    mask = pd.Series(True, index=df.index)
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        mask &= df[col].between(lower, upper)
    n_removed = (~mask).sum()
    logger.info("Removed %d outlier rows via IQR method across %d columns", n_removed, len(columns))
    return df.loc[mask].reset_index(drop=True)


def remove_outliers_zscore(df: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> pd.DataFrame:
    """Remove rows with |z-score| > threshold in any of the given columns.

    Args:
        df: Input DataFrame.
        columns: Numeric columns to check.
        threshold: Z-score cutoff.

    Returns:
        DataFrame with outlier rows removed.
    """
    z = (df[columns] - df[columns].mean()) / df[columns].std(ddof=0)
    mask = (z.abs() <= threshold).all(axis=1)
    logger.info("Removed %d outlier rows via z-score method", (~mask).sum())
    return df.loc[mask].reset_index(drop=True)


def encode_categoricals(df: pd.DataFrame, method: str = "onehot") -> pd.DataFrame:
    """One-hot (or label) encode categorical columns.

    Args:
        df: Input DataFrame.
        method: "onehot" or "label".

    Returns:
        Encoded DataFrame.
    """
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if not cat_cols:
        return df

    if method == "onehot":
        return pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes
    return df


def drop_correlated_features(df: pd.DataFrame, target_col: str, threshold: float = 0.90) -> pd.DataFrame:
    """Drop one of each pair of features correlated above ``threshold``.

    The target column is excluded from pruning consideration.

    Args:
        df: Input DataFrame (numeric features + target).
        target_col: Name of the target column to protect from removal.
        threshold: Absolute correlation above which one feature is dropped.

    Returns:
        DataFrame with redundant, highly-correlated features removed.
    """
    feature_df = df.drop(columns=[target_col])
    corr = feature_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    logger.info("Dropping %d highly correlated features (>%.2f): %s", len(to_drop), threshold, to_drop)
    return df.drop(columns=to_drop)


def compute_vif(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each feature.

    Args:
        df: DataFrame of numeric features.
        feature_cols: Columns to include in the VIF calculation.

    Returns:
        DataFrame with columns ["feature", "VIF"], sorted descending. VIF>10
        indicates problematic multicollinearity.
    """
    X = df[feature_cols].astype(float).values
    vif_data = pd.DataFrame({
        "feature": feature_cols,
        "VIF": [variance_inflation_factor(X, i) for i in range(len(feature_cols))],
    })
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                    method: str = "standard") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a scaler on train and apply to train/val/test.

    Args:
        X_train, X_val, X_test: Feature splits.
        method: "standard" (StandardScaler, for linear models) or
            "minmax" (MinMaxScaler).

    Returns:
        Tuple of scaled (X_train, X_val, X_test) as DataFrames with
        original column names/index preserved.
    """
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_s, X_val_s, X_test_s


def split_data(df: pd.DataFrame, target_col: str, config: dict,
                stratify_col: str = None) -> dict:
    """Split into 70/15/15 train/val/test.

    Args:
        df: Full engineered+cleaned DataFrame.
        target_col: Name of target column.
        config: Project config (uses preprocessing.train/val/test size).
        stratify_col: Optional categorical column (e.g. BMI_Category) to
            stratify the split on.

    Returns:
        Dict with keys X_train, X_val, X_test, y_train, y_val, y_test.
    """
    seed = config["random_seed"]
    train_size = config["preprocessing"]["train_size"]
    val_size = config["preprocessing"]["val_size"]
    test_size = config["preprocessing"]["test_size"]

    y = df[target_col]
    X = df.drop(columns=[target_col])
    strat = df[stratify_col] if stratify_col and stratify_col in df.columns else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, random_state=seed, stratify=strat
    )
    strat_temp = strat.loc[X_temp.index] if strat is not None else None
    relative_val = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=relative_val, random_state=seed, stratify=strat_temp
    )

    logger.info("Split sizes -> train: %d, val: %d, test: %d", len(X_train), len(X_val), len(X_test))
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }


def run_full_preprocessing(df: pd.DataFrame, target_col: str, config: dict) -> dict:
    """Execute the complete preprocessing pipeline end to end.

    Steps: drop high-missing columns -> impute -> outlier removal ->
    encode categoricals -> drop correlated features -> split -> scale.

    Args:
        df: Feature-engineered DataFrame (output of feature_engineering.py).
        target_col: Name of the RMR target column.
        config: Project configuration.

    Returns:
        Dict containing the split/scaled data plus intermediate artifacts
        (e.g. VIF table) for reporting.
    """
    cfg = config["preprocessing"]

    df = drop_high_missing_columns(df, cfg["missing_value_drop_threshold"])
    df = df.dropna(subset=[target_col])  # target must be present
    df = impute_missing(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors="ignore").tolist()
    if cfg["outlier_method"] == "iqr":
        df = remove_outliers_iqr(df, numeric_cols, cfg["iqr_multiplier"])
    else:
        df = remove_outliers_zscore(df, numeric_cols, cfg["zscore_threshold"])

    bmi_category = df["BMI_Category"].astype(str) if "BMI_Category" in df.columns else None
    df_encoded = encode_categoricals(df, method="onehot")
    df_encoded = drop_correlated_features(df_encoded, target_col, cfg["correlation_drop_threshold"])

    if bmi_category is not None:
        df_encoded["_strat_bmi_cat"] = bmi_category.values

    splits = split_data(df_encoded, target_col, config,
                         stratify_col="_strat_bmi_cat" if bmi_category is not None else None)

    for key in ["X_train", "X_val", "X_test"]:
        splits[key] = splits[key].drop(columns=["_strat_bmi_cat"], errors="ignore")

    feature_cols = splits["X_train"].select_dtypes(include=[np.number]).columns.tolist()
    try:
        vif_table = compute_vif(splits["X_train"], feature_cols)
    except Exception as exc:  # VIF can fail on perfectly collinear dummy columns
        logger.warning("VIF computation failed: %s", exc)
        vif_table = pd.DataFrame(columns=["feature", "VIF"])

    X_train_std, X_val_std, X_test_std = scale_features(
        splits["X_train"], splits["X_val"], splits["X_test"], method="standard"
    )
    X_train_mm, X_val_mm, X_test_mm = scale_features(
        splits["X_train"], splits["X_val"], splits["X_test"], method="minmax"
    )

    splits.update({
        "X_train_standard": X_train_std, "X_val_standard": X_val_std, "X_test_standard": X_test_std,
        "X_train_minmax": X_train_mm, "X_val_minmax": X_val_mm, "X_test_minmax": X_test_mm,
        "vif_table": vif_table,
    })
    return splits
