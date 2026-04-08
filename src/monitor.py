"""
Prediction logging and simple data-drift checks versus training reference.
"""
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "data" / "prediction_log.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "X_train.csv"


def log_prediction(
    input_dict: dict,
    prediction: int,
    probability: float,
    log_path: Path | None = None,
) -> None:
    """
    Append one prediction row with timestamp and inputs to CSV.

    Args:
        input_dict: Feature name -> value for the applicant row.
        prediction: Predicted class (0 or 1).
        probability: Predicted positive-class probability.
        log_path: CSV path; defaults to data/prediction_log.csv.
    """
    path = Path(log_path) if log_path is not None else LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now(timezone.utc).isoformat()}
    row.update(input_dict)
    row["predicted_class"] = prediction
    row["probability"] = probability
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)


def compute_drift_score(
    new_data_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compare column means to reference; flag large mean shifts.

    Args:
        new_data_df: Recent batch of feature rows.
        reference_df: Training reference (e.g. X_train).

    Returns:
        Tuple of summary DataFrame and list of flagged feature names.
    """
    num_cols = new_data_df.select_dtypes(include=["number"]).columns
    flags = []
    rows = []
    ref = reference_df[num_cols]
    new = new_data_df[num_cols]
    for c in num_cols:
        m0 = float(ref[c].mean())
        s0 = float(ref[c].std(ddof=0)) or 1e-9
        m1 = float(new[c].mean())
        z = abs(m1 - m0) / s0
        flagged = z > 2.0
        if flagged:
            flags.append(c)
        rows.append(
            {
                "feature": c,
                "ref_mean": m0,
                "ref_std": s0,
                "new_mean": m1,
                "z_score_mean_diff": z,
                "flagged": flagged,
            }
        )
    return pd.DataFrame(rows), flags


def check_and_alert(
    new_data_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> bool:
    """
    Print drift alerts and recommend retraining if any feature drifts.

    Args:
        new_data_df: Recent feature batch.
        reference_df: Training reference distribution.

    Returns:
        True if retraining is recommended (any flagged feature).
    """
    summary, flags = compute_drift_score(new_data_df, reference_df)
    for c in flags:
        print(f"DRIFT ALERT: feature '{c}' mean differs >2 ref std from training.")
    if flags:
        print("Retraining recommended: incoming distribution shifted.")
        return True
    print("No significant drift detected.")
    return False


def main() -> None:
    """Load logs and training data; print drift monitoring report."""
    if not LOG_PATH.exists():
        print(f"No prediction log at {LOG_PATH}; nothing to compare.")
        return
    if not TRAIN_PATH.exists():
        print(f"Missing reference training data at {TRAIN_PATH}.")
        return
    log_df = pd.read_csv(LOG_PATH)
    ref = pd.read_csv(TRAIN_PATH)
    feature_cols = [c for c in ref.columns if c in log_df.columns]
    if not feature_cols:
        print("No overlapping feature columns between log and training data.")
        return
    recent = log_df[feature_cols].tail(500)
    summary, _ = compute_drift_score(recent, ref)
    print("Drift monitoring report (recent batch vs training):")
    print(summary.to_string())
    check_and_alert(recent, ref)


if __name__ == "__main__":
    main()
