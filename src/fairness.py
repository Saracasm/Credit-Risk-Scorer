"""
Fairlearn demographic audit by age group on the test set.
"""
from pathlib import Path

import joblib
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_pipeline.pkl"
FAIRNESS_REPORT_PATH = PROJECT_ROOT / "data" / "fairness_report.csv"


def _age_bins(ages: pd.Series) -> pd.Series:
    """
    Map age values to four bucket labels.

    Args:
        ages: Series of ages.

    Returns:
        Categorical age group labels.
    """
    return pd.cut(
        ages,
        bins=[17, 30, 45, 60, 120],
        labels=["18-30", "31-45", "46-60", "60+"],
    )


def main() -> None:
    """Compute per-group metrics, print disparity, save CSV."""
    pipe = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(SPLITS_DIR / "X_test.csv")
    y_test = pd.read_csv(SPLITS_DIR / "y_test.csv").squeeze()
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    y_pred = (pipe.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    groups = _age_bins(X_test["age"])

    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "precision": lambda yt, yp: precision_score(
                yt, yp, zero_division=0
            ),
            "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=groups,
    )

    report = mf.by_group.copy()
    FAIRNESS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(FAIRNESS_REPORT_PATH)

    print("Per-group metrics:")
    print(report)

    for col in report.columns:
        vals = report[col].astype(float)
        mx, mn = float(vals.max()), float(vals.min())
        disp = mx - mn
        print(f"Max disparity ({col}): {disp:.4f} (best {mx:.4f}, worst {mn:.4f})")

    worst_acc = report["accuracy"].idxmin()
    print(
        f"\nInterpretation: lowest accuracy is in group '{worst_acc}' - "
        "monitor decisions for this segment and consider calibration or "
        "additional features."
    )


if __name__ == "__main__":
    main()
