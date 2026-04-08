"""
Exploratory data analysis for Give Me Some Credit (run once).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "cs-training.csv"


def main() -> None:
    """Load data, print summaries, and save EDA figures."""
    if not DATA_PATH.exists():
        print(f"Missing {DATA_PATH}. Download cs-training.csv from Kaggle into data/.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    print("=== shape ===")
    print(df.shape)
    print("\n=== dtypes ===")
    print(df.dtypes)
    print("\n=== head ===")
    print(df.head())
    print("\n=== describe ===")
    print(df.describe().T)

    miss = df.isnull().sum()
    miss_pct = 100 * miss / len(df)
    print("\n=== missing ===")
    print(pd.DataFrame({"count": miss, "pct": miss_pct}))

    target = "SeriousDlqin2yrs"
    vc = df[target].value_counts()
    print("\n=== target distribution ===")
    print(vc)
    print(vc / len(df))

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    n = len(df.columns)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
    axes = np.array(axes).flatten()
    for i, c in enumerate(df.columns):
        axes[i].hist(df[c].dropna(), bins=50, edgecolor="black", alpha=0.7)
        axes[i].set_title(c, fontsize=8)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(data_dir / "distributions.png", dpi=120)
    plt.close()

    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(data_dir / "correlation_matrix.png", dpi=120)
    plt.close()

    df["age_bin"] = pd.cut(df["age"], bins=[0, 30, 45, 60, 120], labels=["0-30", "31-45", "46-60", "60+"])
    rate = df.groupby("age_bin", observed=False)[target].mean()
    plt.figure(figsize=(8, 5))
    rate.plot(kind="bar", color="steelblue")
    plt.ylabel("Default rate")
    plt.xlabel("Age group")
    plt.title("Default rate by age group")
    plt.tight_layout()
    plt.savefig(data_dir / "default_by_age.png", dpi=120)
    plt.close()

    num_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(12, 8))
    df[num_cols].boxplot(rot=90)
    plt.title("Outlier boxplots (numeric columns)")
    plt.tight_layout()
    plt.savefig(data_dir / "outlier_boxplots.png", dpi=120)
    plt.close()

    print("\n=== EDA summary ===")
    print(
        "- Missing: MonthlyIncome and NumberOfDependents have missing values.\n"
        "- Target: strong class imbalance (~93% non-default).\n"
        "- Skew / outliers: RevolvingUtilization, DebtRatio, late-payment counts have heavy tails.\n"
        "- Age has invalid zeros; utilization and debt ratio need clipping.\n"
    )
    print(f"Figures saved under {data_dir}")


if __name__ == "__main__":
    main()
