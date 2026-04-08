"""
Data cleaning for the Give Me Some Credit dataset.
"""
from pathlib import Path

import pandas as pd

# Default paths (overridable via function args)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "cs-training.csv"
DEFAULT_CLEANED_PATH = PROJECT_ROOT / "data" / "cs-cleaned.csv"

LATE_PAYMENT_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


def load_raw(path: Path | str) -> pd.DataFrame:
    """
    Load the raw Kaggle CSV and drop the index column if present.

    Args:
        path: Filesystem path to cs-training.csv.

    Returns:
        DataFrame of raw training data.
    """
    p = Path(path)
    df = pd.read_csv(p)
    # Kaggle export often includes Unnamed: 0
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    print(f"load_raw: shape={df.shape}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame without duplicate rows.
    """
    before = len(df)
    out = df.drop_duplicates()
    removed = before - len(out)
    print(f"remove_duplicates: removed {removed} duplicate rows")
    return out


def fix_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where age is outside plausible borrower range.

    Args:
        df: Input DataFrame with an 'age' column.

    Returns:
        DataFrame with underage/overage rows removed.
    """
    before = len(df)
    out = df[(df["age"] >= 18) & (df["age"] <= 110)].copy()
    removed = before - len(out)
    print(f"fix_impossible_values: removed {removed} rows (age < 18 or age > 110)")
    return out


def fix_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing MonthlyIncome and NumberOfDependents with column medians.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with missing values filled.
    """
    out = df.copy()
    inc_med = out["MonthlyIncome"].median()
    dep_med = out["NumberOfDependents"].median()
    out["MonthlyIncome"] = out["MonthlyIncome"].fillna(inc_med)
    out["NumberOfDependents"] = out["NumberOfDependents"].fillna(dep_med)
    print(
        "fix_missing_values: filled MonthlyIncome with median "
        f"{inc_med}, NumberOfDependents with median {dep_med}"
    )
    return out


def fix_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip extreme values for utilization, debt ratio, income, and late counts.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with clipped numeric columns.
    """
    out = df.copy()
    out["RevolvingUtilizationOfUnsecuredLines"] = out[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(0, 1)
    out["DebtRatio"] = out["DebtRatio"].clip(0, 10)
    inc_99 = out["MonthlyIncome"].quantile(0.99)
    out["MonthlyIncome"] = out["MonthlyIncome"].clip(upper=inc_99)
    for col in LATE_PAYMENT_COLS:
        out[col] = out[col].clip(0, 20)
    print("fix_outliers: clipped utilization, DebtRatio, income 99th pct, late counts")
    return out


def clean(path: Path | str | None = None) -> pd.DataFrame:
    """
    Run the full cleaning pipeline on the raw CSV path.

    Args:
        path: Path to cs-training.csv. Defaults to data/cs-training.csv.

    Returns:
        Cleaned DataFrame.
    """
    raw_path = Path(path) if path is not None else DEFAULT_RAW_PATH
    df = load_raw(raw_path)
    df = remove_duplicates(df)
    df = fix_impossible_values(df)
    df = fix_missing_values(df)
    df = fix_outliers(df)
    return df


if __name__ == "__main__":
    cleaned = clean()
    assert cleaned.isnull().sum().sum() == 0, "missing values remain after clean"
    DEFAULT_CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(DEFAULT_CLEANED_PATH, index=False)
    print(f"Saved cleaned data to {DEFAULT_CLEANED_PATH}")
