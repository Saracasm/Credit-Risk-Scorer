"""
Feature engineering and train/test preparation for credit risk modeling.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_PATH = PROJECT_ROOT / "data" / "cs-cleaned.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
SCALER_PATH = MODELS_DIR / "scaler.pkl"


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add six engineered columns used by the model.

    Args:
        X: Feature DataFrame (no target column).

    Returns:
        DataFrame with six additional columns.
    """
    out = X.copy()
    out["total_late_payments"] = (
        out["NumberOfTime30-59DaysPastDueNotWorse"]
        + out["NumberOfTime60-89DaysPastDueNotWorse"]
        + out["NumberOfTimes90DaysLate"]
    )
    out["ever_90_days_late"] = (out["NumberOfTimes90DaysLate"] > 0).astype(int)
    out["monthly_debt_estimate"] = out["DebtRatio"] * out["MonthlyIncome"]
    out["income_per_dependent"] = out["MonthlyIncome"] / (
        out["NumberOfDependents"] + 1
    )
    out["age_group"] = (
        pd.cut(
            out["age"],
            bins=[0, 30, 45, 60, 120],
            labels=[0, 1, 2, 3],
            include_lowest=True,
        )
        .astype(float)
    )
    out["high_utilization"] = (
        out["RevolvingUtilizationOfUnsecuredLines"] > 0.9
    ).astype(int)
    return out


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate target from features and apply engineered features to X only.

    Args:
        df: Cleaned DataFrame including SeriousDlqin2yrs.

    Returns:
        Tuple (X, y) with engineered features in X.
    """
    y = df["SeriousDlqin2yrs"].copy()
    X = df.drop(columns=["SeriousDlqin2yrs"])
    X = add_engineered_features(X)
    rate = y.mean()
    print(f"split_features_target: X.shape={X.shape}, y.shape={y.shape}, default rate={rate:.4f}")
    return X, y


def split_train_test(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Fraction for test set.
        random_state: RNG seed.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler on training features only; transform train and test.

    Args:
        X_train: Training features.
        X_test: Test features.

    Returns:
        Scaled X_train, scaled X_test, fitted scaler.
    """
    scaler = StandardScaler()
    cols = X_train.columns
    X_tr = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=cols,
        index=X_train.index,
    )
    X_te = pd.DataFrame(
        scaler.transform(X_test),
        columns=cols,
        index=X_test.index,
    )
    return X_tr, X_te, scaler


def load_cleaned(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load cleaned CSV produced by preprocess.clean.

    Args:
        path: Path to cs-cleaned.csv.

    Returns:
        DataFrame.
    """
    p = Path(path) if path is not None else CLEANED_PATH
    return pd.read_csv(p)


if __name__ == "__main__":
    import joblib

    df = load_cleaned()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    _, _, scaler = scale_features(X_train, X_test)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save unscaled engineered features for Pipeline training (scaler inside model)
    X_train.to_csv(SPLITS_DIR / "X_train.csv", index=False)
    X_test.to_csv(SPLITS_DIR / "X_test.csv", index=False)
    y_train.to_csv(SPLITS_DIR / "y_train.csv", index=False)
    y_test.to_csv(SPLITS_DIR / "y_test.csv", index=False)

    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved splits to {SPLITS_DIR}, scaler to {SCALER_PATH}")
