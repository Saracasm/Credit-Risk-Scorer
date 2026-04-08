"""
Single-row inference helpers for the Streamlit app and tests.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import add_engineered_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_pipeline.pkl"


def load_pipeline(path: Path | None = None):
    """
    Load the trained sklearn Pipeline from disk.

    Args:
        path: Path to joblib file; defaults to models/xgb_pipeline.pkl.

    Returns:
        Fitted Pipeline with scaler and XGBoost.
    """
    p = Path(path) if path is not None else MODEL_PATH
    return joblib.load(p)


def build_feature_row(
    age: float,
    monthly_income: float,
    debt_ratio: float,
    revolving_util: float,
    n_30_59_late: float,
    n_60_89_late: float,
    n_90_late: float,
    n_open_lines: float,
    n_real_estate: float,
    n_dependents: float,
) -> pd.DataFrame:
    """
    Build one engineered feature row matching training columns.

    Args:
        Raw feature values as named in the Kaggle dataset.

    Returns:
        Single-row DataFrame with engineered features.
    """
    row = {
        "RevolvingUtilizationOfUnsecuredLines": revolving_util,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": n_30_59_late,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income,
        "NumberOfOpenCreditLinesAndLoans": n_open_lines,
        "NumberOfTimes90DaysLate": n_90_late,
        "NumberRealEstateLoansOrLines": n_real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": n_60_89_late,
        "NumberOfDependents": n_dependents,
    }
    df = pd.DataFrame([row])
    return add_engineered_features(df)


def predict_default(
    pipe,
    X: pd.DataFrame,
) -> tuple[float, int]:
    """
    Return positive-class probability and thresholded class.

    Args:
        pipe: Trained Pipeline.
        X: Feature DataFrame (one or more rows).

    Returns:
        Tuple (probability for first row, predicted class for first row at 0.5).
    """
    proba = pipe.predict_proba(X)[:, 1]
    p = float(proba[0])
    cls = int(p >= 0.5)
    return p, cls


def shap_values_row(pipe, X: pd.DataFrame):
    """
    Compute SHAP values for the first row using TreeExplainer.

    Args:
        pipe: Trained Pipeline.
        X: Single-row engineered features.

    Returns:
        Tuple (feature names, shap values for positive class, base value).
    """
    import shap

    scaler = pipe.named_steps["scaler"]
    model = pipe.named_steps["model"]
    Xt = scaler.transform(X)
    explainer = shap.TreeExplainer(model)
    sv = explainer(Xt)
    vals_arr = np.asarray(sv.values)
    base_arr = np.asarray(sv.base_values)
    if vals_arr.ndim == 3:
        vals = vals_arr[0, :, 1]
        base = base_arr[0, 1] if base_arr.ndim == 2 else base_arr[1]
    else:
        vals = vals_arr[0]
        base = base_arr[0] if base_arr.ndim else base_arr
    names = list(X.columns)
    return names, np.asarray(vals).ravel(), float(np.asarray(base).ravel()[0])
