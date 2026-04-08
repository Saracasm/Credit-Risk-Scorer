"""
Compare baseline models with stratified cross-validation and MLflow logging.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

try:
    import mlflow

    mlflow.set_experiment("credit-risk-scorer")
    _HAS_MLFLOW = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    _HAS_MLFLOW = False


def _make_cv() -> StratifiedKFold:
    """Return a 5-fold stratified splitter with fixed seed."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def _build_lr_pipeline() -> ImbPipeline:
    """Logistic regression with scaling, SMOTE, and balanced classes."""
    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )
    # k_neighbors=1 avoids SMOTE failures when a CV fold has very few minority points
    return ImbPipeline(
        [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42, k_neighbors=1)),
            ("clf", lr),
        ]
    )


def _build_rf_pipeline() -> ImbPipeline:
    """Random forest with scaling, SMOTE, and balanced classes."""
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    return ImbPipeline(
        [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42, k_neighbors=1)),
            ("clf", rf),
        ]
    )


def _build_xgb_pipeline(scale_pos_weight: float) -> ImbPipeline:
    """XGBoost with scaling, SMOTE, and class-weight ratio."""
    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
    )
    return ImbPipeline(
        [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42, k_neighbors=1)),
            ("clf", xgb),
        ]
    )


def _run_cv(
    name: str, model_type: str, pipeline: ImbPipeline, X: pd.DataFrame, y: pd.Series
) -> dict:
    """
    Run cross-validation and log metrics to MLflow.

    Args:
        name: MLflow run name.
        model_type: Short label for params.
        pipeline: imblearn Pipeline.
        X: Training features.
        y: Training labels.

    Returns:
        Dict of mean/std roc_auc and mean precision/recall.
    """
    cv = _make_cv()
    scoring = ["roc_auc", "precision", "recall"]
    out = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    roc_mean = float(np.mean(out["test_roc_auc"]))
    roc_std = float(np.std(out["test_roc_auc"]))
    pr_mean = float(np.mean(out["test_precision"]))
    rec_mean = float(np.mean(out["test_recall"]))

    if _HAS_MLFLOW and mlflow is not None:
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", model_type)
            mlflow.log_metric("cv_roc_auc_mean", roc_mean)
            mlflow.log_metric("cv_roc_auc_std", roc_std)
            mlflow.log_metric("cv_precision_mean", pr_mean)
            mlflow.log_metric("cv_recall_mean", rec_mean)
    else:
        print(
            f"[no mlflow] {name}: auc={roc_mean:.4f}±{roc_std:.4f} "
            f"prec={pr_mean:.4f} rec={rec_mean:.4f}"
        )

    return {
        "roc_mean": roc_mean,
        "roc_std": roc_std,
        "precision_mean": pr_mean,
        "recall_mean": rec_mean,
    }


def main() -> None:
    """Load splits, compare three models, print ranking."""
    X_train = pd.read_csv(SPLITS_DIR / "X_train.csv")
    y_train = pd.read_csv(SPLITS_DIR / "y_train.csv").squeeze()
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = float(neg / pos)

    results = []
    r_lr = _run_cv(
        "logistic_regression", "LogisticRegression", _build_lr_pipeline(), X_train, y_train
    )
    results.append(("LogisticRegression", r_lr["roc_mean"], r_lr))
    r_rf = _run_cv(
        "random_forest", "RandomForest", _build_rf_pipeline(), X_train, y_train
    )
    results.append(("RandomForest", r_rf["roc_mean"], r_rf))
    r_xgb = _run_cv(
        "xgboost_baseline", "XGBoost", _build_xgb_pipeline(spw), X_train, y_train
    )
    results.append(("XGBoost", r_xgb["roc_mean"], r_xgb))

    results.sort(key=lambda x: x[1], reverse=True)
    print("\n=== Model comparison (by ROC-AUC mean) ===")
    for name, roc, _ in results:
        print(f"  {name}: {roc:.4f}")

    best_name, best_roc, _ = results[0]
    print(
        f"\nSelected model: {best_name} with ROC-AUC of {best_roc:.3f}"
    )


if __name__ == "__main__":
    main()
