"""
Unit tests for preprocessing helpers.
"""
from pathlib import Path

import pytest

from src.preprocess import clean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "cs-training.csv"
FIXTURE_CSV = Path(__file__).resolve().parent / "fixtures" / "cs-training-sample.csv"


def _raw_path() -> Path:
    """Prefer real Kaggle file; fall back to committed sample."""
    if DATA_CSV.exists():
        return DATA_CSV
    if FIXTURE_CSV.exists():
        return FIXTURE_CSV
    pytest.skip("No cs-training.csv or tests/fixtures/cs-training-sample.csv")


def test_no_missing_after_clean() -> None:
    """After clean(), there must be no nulls."""
    df = clean(_raw_path())
    assert df.isnull().sum().sum() == 0


def test_no_underage_rows() -> None:
    """All ages must be at least 18 after cleaning."""
    df = clean(_raw_path())
    assert (df["age"] >= 18).all()


def test_outlier_clipping() -> None:
    """Revolving utilization must be clipped to [0, 1]."""
    df = clean(_raw_path())
    assert df["RevolvingUtilizationOfUnsecuredLines"].max() <= 1.0


def test_no_duplicates() -> None:
    """No duplicate rows after cleaning."""
    df = clean(_raw_path())
    assert not df.duplicated().any()
