"""
Model evaluation: metrics, plots, and SHAP explanations.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
PLOTS_DIR = PROJECT_ROOT / "data" / "plots"
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_pipeline.pkl"


def _load_artifacts() -> tuple:
    """Load pipeline and test sets."""
    pipe = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(SPLITS_DIR / "X_test.csv")
    y_test = pd.read_csv(SPLITS_DIR / "y_test.csv").squeeze()
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
    return pipe, X_test, y_test


def _plot_roc(y_true, y_score, path: Path) -> None:
    """Save ROC curve PNG."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_pr(y_true, y_score, path: Path) -> None:
    """Save precision-recall curve PNG."""
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"PR (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_confusion(y_true, y_pred, path: Path) -> None:
    """Save confusion matrix heatmap at threshold 0.5."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_thresholds(y_true, y_score, path: Path) -> None:
    """Plot precision, recall, F1 vs threshold."""
    thresholds = np.linspace(0.1, 0.9, 81)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        precs.append(precision_score(y_true, pred, zero_division=0))
        recs.append(recall_score(y_true, pred, zero_division=0))
        f1s.append(f1_score(y_true, pred, zero_division=0))
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precs, label="Precision")
    plt.plot(thresholds, recs, label="Recall")
    plt.plot(thresholds, f1s, label="F1")
    plt.xlabel("Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _shap_values_for_positive_class(sv) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract SHAP values and base values for the positive (default) class.

    Args:
        sv: shap.Explanation from TreeExplainer.

    Returns:
        Tuple (values_2d [n_samples, n_features], base_values_1d per sample).
    """
    vals = np.asarray(sv.values)
    base = np.asarray(sv.base_values)
    if vals.ndim == 3:
        # Binary: (n_samples, n_features, 2) — use class 1 (default)
        vals = vals[:, :, 1]
        base = base[:, 1] if base.ndim == 2 else base
    return vals, base


def _shap_plots(pipe, X_test: pd.DataFrame) -> None:
    """Generate SHAP summary, bar, and waterfall plots for first 5 rows."""
    scaler = pipe.named_steps["scaler"]
    model = pipe.named_steps["model"]
    Xs = scaler.transform(X_test)
    cols = list(X_test.columns)
    Xs_df = pd.DataFrame(Xs, columns=cols)

    explainer = shap.TreeExplainer(model)
    sv = explainer(Xs)
    vals2d, base_per_row = _shap_values_for_positive_class(sv)

    plt.figure()
    shap.summary_plot(vals2d, Xs_df, show=False, plot_type="dot")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(vals2d, Xs_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close()

    base_arr = np.asarray(base_per_row)
    for i in range(min(5, len(X_test))):
        if base_arr.ndim == 0 or base_arr.size == 1:
            b = float(base_arr.ravel()[0])
        else:
            b = float(base_arr[i])
        exp = shap.Explanation(
            values=vals2d[i],
            base_values=b,
            data=Xs_df.iloc[i].values,
            feature_names=cols,
        )
        plt.figure()
        shap.plots.waterfall(exp, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"shap_force_{i}.png", dpi=120, bbox_inches="tight")
        plt.close()


def main() -> None:
    """Produce all evaluation plots under data/plots/."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pipe, X_test, y_test = _load_artifacts()
    y_score = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    _plot_roc(y_test, y_score, PLOTS_DIR / "roc_curve.png")
    _plot_pr(y_test, y_score, PLOTS_DIR / "pr_curve.png")
    _plot_confusion(y_test, y_pred, PLOTS_DIR / "confusion_matrix.png")
    _plot_thresholds(y_test, y_score, PLOTS_DIR / "threshold_analysis.png")

    print(f"ROC-AUC (test): {roc_auc_score(y_test, y_score):.4f}")
    _shap_plots(pipe, X_test)
    print(f"Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
