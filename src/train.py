"""
Optuna tuning for XGBoost and save sklearn Pipeline (scaler + model).
"""
from pathlib import Path

import joblib
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_pipeline.pkl"

try:
    import mlflow

    mlflow.set_experiment("credit-risk-scorer")
    _HAS_MLFLOW = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    _HAS_MLFLOW = False

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load_train() -> tuple[pd.DataFrame, pd.Series]:
    """Load training features and labels from split CSVs."""
    X = pd.read_csv(SPLITS_DIR / "X_train.csv")
    y = pd.read_csv(SPLITS_DIR / "y_train.csv").squeeze()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    return X, y


def _make_objective(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
):
    """
    Build Optuna objective that maximizes ROC-AUC on validation fold.

    Args:
        X, y: Training fold.
        X_val, y_val: Validation fold.
        scale_pos_weight: XGBoost imbalance ratio.

    Returns:
        Callable trial -> roc_auc score.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(X, y)
        proba = pipe.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_val, proba))

    return objective


def main() -> None:
    """Tune hyperparameters, fit final pipeline, save artifact and MLflow."""
    X, y = _load_train()
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = float(neg / pos)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(
        _make_objective(X_tr, y_tr, X_val, y_val, spw),
        n_trials=50,
        show_progress_bar=True,
    )
    best = study.best_params

    final_model = XGBClassifier(
        **best,
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", final_model)])
    pipeline.fit(X, y)

    X_test = pd.read_csv(SPLITS_DIR / "X_test.csv")
    y_test = pd.read_csv(SPLITS_DIR / "y_test.csv").squeeze()
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
    test_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    if _HAS_MLFLOW and mlflow is not None:
        with mlflow.start_run(run_name="xgb_optuna_final"):
            for k, v in best.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("test_roc_auc", float(test_auc))
            mlflow.log_param("scale_pos_weight", spw)
    else:
        print(f"[no mlflow] test_roc_auc={test_auc:.4f} scale_pos_weight={spw}")

    print(f"Best params: {best}")
    print(f"Test ROC-AUC: {test_auc:.4f}")
    print(f"Saved pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
