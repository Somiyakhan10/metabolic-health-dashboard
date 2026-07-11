"""
feature_engineering.py
=======================
Creates the RMR target (Mifflin-St Jeor equation) and engineers clinically
meaningful predictor features from merged NHANES data.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_rmr_mifflin_st_jeor(df: pd.DataFrame) -> pd.Series:
    """Compute Resting Metabolic Rate via the Mifflin-St Jeor equation."""
    required = ["RIAGENDR", "BMXWT", "BMXHT", "RIDAGEYR"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns for RMR calculation: {missing_cols}")

    base = 10 * df["BMXWT"] + 6.25 * df["BMXHT"] - 5 * df["RIDAGEYR"]
    sex_offset = np.where(df["RIAGENDR"] == 1, 5, -161)
    rmr = base + sex_offset

    mask_valid = df[required].notna().all(axis=1)
    rmr = rmr.where(mask_valid, np.nan)
    return rmr.rename("RMR_kcal_day")


def compute_homa_ir(df: pd.DataFrame,
                     glucose_col: str = "LBXGLU",
                     insulin_col: str = "LBXIN") -> pd.Series:
    """Compute HOMA-IR (insulin resistance index)."""
    if glucose_col not in df.columns or insulin_col not in df.columns:
        logger.warning("HOMA-IR inputs missing; returning all-NaN column")
        return pd.Series(np.nan, index=df.index, name="HOMA_IR")

    homa_ir = (df[glucose_col] * df[insulin_col]) / 405.0
    return homa_ir.rename("HOMA_IR")


def compute_waist_to_height_ratio(df: pd.DataFrame,
                                   waist_col: str = "BMXWAIST",
                                   height_col: str = "BMXHT") -> pd.Series:
    """Compute waist-to-height ratio (WHtR)."""
    if waist_col not in df.columns or height_col not in df.columns:
        logger.warning("Waist or height columns missing; returning all-NaN column")
        return pd.Series(np.nan, index=df.index, name="Waist_to_Height_Ratio")
    return (df[waist_col] / df[height_col]).rename("Waist_to_Height_Ratio")


def flag_metabolic_syndrome(df: pd.DataFrame, config: dict) -> pd.Series:
    """Flag metabolic syndrome using AHA/NHLBI (ATP III) criteria."""
    try:
        crit = config["feature_engineering"]["metabolic_syndrome_criteria"]
    except KeyError:
        logger.warning("Metabolic syndrome criteria not found; returning all-NaN")
        return pd.Series(np.nan, index=df.index, name="Metabolic_Syndrome_Flag")

    is_male = df["RIAGENDR"] == 1
    waist_cutoff = np.where(is_male, crit["waist_cm_male"], crit["waist_cm_female"])
    hdl_cutoff = np.where(is_male, crit["hdl_male_mgdl"], crit["hdl_female_mgdl"])

    waist = df.get("BMXWAIST", pd.Series(np.nan, index=df.index))
    trig = df.get("LBXTR", pd.Series(np.nan, index=df.index))
    hdl = df.get("LBDHDD", pd.Series(np.nan, index=df.index))
    sbp = df.get("BPXSY1", pd.Series(np.nan, index=df.index))
    dbp = df.get("BPXDI1", pd.Series(np.nan, index=df.index))
    glucose = df.get("LBXGLU", pd.Series(np.nan, index=df.index))

    c1 = waist if isinstance(waist, pd.Series) else pd.Series(waist, index=df.index)
    c2 = trig if isinstance(trig, pd.Series) else pd.Series(trig, index=df.index)
    c3 = hdl if isinstance(hdl, pd.Series) else pd.Series(hdl, index=df.index)
    c4 = sbp if isinstance(sbp, pd.Series) else pd.Series(sbp, index=df.index)
    c5 = dbp if isinstance(dbp, pd.Series) else pd.Series(dbp, index=df.index)
    c6 = glucose if isinstance(glucose, pd.Series) else pd.Series(glucose, index=df.index)

    c1_flag = c1 >= waist_cutoff
    c2_flag = c2 >= crit["triglycerides_mgdl"]
    c3_flag = c3 < hdl_cutoff
    c4_flag = (c4 >= 130) | (c5 >= 85)
    c5_flag = c6 >= crit["fasting_glucose_mgdl"]

    criteria_matrix = pd.concat([c1_flag, c2_flag, c3_flag, c4_flag, c5_flag], axis=1)
    n_criteria = criteria_matrix.sum(axis=1, skipna=True)
    n_available = criteria_matrix.notna().sum(axis=1)

    flag = (n_criteria >= 3).astype(float)
    flag = flag.where(n_available >= 3, np.nan)
    return flag.rename("Metabolic_Syndrome_Flag")


def categorize_physical_activity(df: pd.DataFrame) -> pd.Series:
    """Bucket participants into Sedentary / Moderate / Vigorous activity."""
    vigorous = df.get("PAQ650")
    moderate = df.get("PAQ665")

    if vigorous is None or moderate is None:
        logger.info("Physical activity columns not found; setting all to 'Unknown'")
        return pd.Series(["Unknown"] * len(df), index=df.index, name="Physical_Activity_Category")

    def _classify(v, m):
        if pd.isna(v) and pd.isna(m):
            return "Unknown"
        if v == 1:
            return "Vigorous"
        if m == 1:
            return "Moderate"
        return "Sedentary"

    return pd.Series(
        [_classify(v, m) for v, m in zip(vigorous, moderate)],
        index=df.index,
        name="Physical_Activity_Category",
    )


def map_diabetes_status(df: pd.DataFrame) -> pd.Series:
    """Map DIQ010 diabetes questionnaire codes to readable labels."""
    diq = df.get("DIQ010")
    if diq is None:
        logger.info("Diabetes status column not found; setting all to 'Unknown'")
        return pd.Series(["Unknown"] * len(df), index=df.index, name="Diabetes_Status")

    mapping = {1: "Diabetic", 2: "Non-diabetic", 3: "Prediabetic"}
    return diq.map(mapping).fillna("Unknown").rename("Diabetes_Status")


def categorize_bmi(df: pd.DataFrame, bmi_col: str = "BMXBMI") -> pd.Series:
    """Bucket BMI into standard WHO categories."""
    if bmi_col not in df.columns:
        logger.warning("BMI column not found; returning all-NaN")
        return pd.Series(np.nan, index=df.index, name="BMI_Category")

    bins = [0, 18.5, 25, 30, np.inf]
    labels = ["Underweight", "Normal", "Overweight", "Obese"]
    return pd.cut(df[bmi_col], bins=bins, labels=labels, right=False).rename("BMI_Category")


def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically motivated interaction terms."""
    df = df.copy()
    if "BMXBMI" in df.columns and "RIDAGEYR" in df.columns:
        df["BMI_x_Age"] = df["BMXBMI"] * df["RIDAGEYR"]
    if "LBXGLU" in df.columns and "LBXGH" in df.columns:
        df["Glucose_x_HbA1c"] = df["LBXGLU"] * df["LBXGH"]
    return df


def engineer_all_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run the full feature-engineering pipeline and append the RMR target."""
    df = df.copy()
    df["RMR_kcal_day"] = compute_rmr_mifflin_st_jeor(df)
    df["HOMA_IR"] = compute_homa_ir(df)
    df["Waist_to_Height_Ratio"] = compute_waist_to_height_ratio(df)
    df["Metabolic_Syndrome_Flag"] = flag_metabolic_syndrome(df, config)
    df["Physical_Activity_Category"] = categorize_physical_activity(df)
    df["Diabetes_Status"] = map_diabetes_status(df)
    df["BMI_Category"] = categorize_bmi(df)
    df = add_interaction_terms(df)

    n_missing_target = df["RMR_kcal_day"].isna().sum()
    logger.info("Feature engineering complete. Rows missing RMR target: %d/%d",
                n_missing_target, len(df))
    return df