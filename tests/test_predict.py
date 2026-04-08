"""
Tests for inference on a valid feature row.
"""
from pathlib import Path

import pytest

from src.inference import (
    build_feature_row,
    load_pipeline,
    predict_default,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_pipeline.pkl"


def _require_model():
    if not MODEL_PATH.exists():
        pytest.skip("Train the model first: python src/train.py")


def test_prediction_output_range() -> None:
    """Predicted probability must lie in [0, 1]."""
    _require_model()
    pipe = load_pipeline()
    X = build_feature_row(
        age=45.0,
        monthly_income=5000.0,
        debt_ratio=0.3,
        revolving_util=0.2,
        n_30_59_late=0.0,
        n_60_89_late=0.0,
        n_90_late=0.0,
        n_open_lines=5.0,
        n_real_estate=1.0,
        n_dependents=1.0,
    )
    prob, _ = predict_default(pipe, X)
    assert 0.0 <= prob <= 1.0


def test_prediction_returns_binary() -> None:
    """Predicted class must be 0 or 1 at 0.5 threshold."""
    _require_model()
    pipe = load_pipeline()
    X = build_feature_row(
        age=35.0,
        monthly_income=4000.0,
        debt_ratio=0.5,
        revolving_util=0.5,
        n_30_59_late=1.0,
        n_60_89_late=0.0,
        n_90_late=0.0,
        n_open_lines=4.0,
        n_real_estate=1.0,
        n_dependents=0.0,
    )
    _, cls = predict_default(pipe, X)
    assert cls in (0, 1)


def test_feature_names_match() -> None:
    """Engineered columns must match training split column names."""
    _require_model()
    import pandas as pd

    ref = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "X_train.csv")
    X = build_feature_row(
        age=40.0,
        monthly_income=6000.0,
        debt_ratio=0.2,
        revolving_util=0.1,
        n_30_59_late=0.0,
        n_60_89_late=0.0,
        n_90_late=0.0,
        n_open_lines=6.0,
        n_real_estate=2.0,
        n_dependents=2.0,
    )
    assert list(X.columns) == list(ref.columns)
