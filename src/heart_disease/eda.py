"""Exploratory data analysis plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from heart_disease.config import REPORTS
from heart_disease.data import load_processed


def run_eda(output_dir: Path | None = None) -> Path:
    output_dir = output_dir or REPORTS
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_processed()
    sns.set_theme(style="whitegrid", palette="muted")

    # Class balance
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["target"].value_counts().sort_index()
    counts.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
    ax.set_title("Class balance (0 = no disease, 1 = disease)")
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    fig.tight_layout()
    balance_path = output_dir / "class_balance.png"
    fig.savefig(balance_path, dpi=120)
    plt.close(fig)

    # Feature histograms
    numeric = [c for c in df.columns if c != "target"]
    n_cols = 4
    n_rows = (len(numeric) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(numeric):
        sns.histplot(df[col], kde=True, ax=axes[i], bins=20)
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature distributions", y=1.02)
    fig.tight_layout()
    hist_path = output_dir / "feature_histograms.png"
    fig.savefig(hist_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature correlation heatmap")
    fig.tight_layout()
    heatmap_path = output_dir / "correlation_heatmap.png"
    fig.savefig(heatmap_path, dpi=120)
    plt.close(fig)

    summary_path = output_dir / "eda_summary.txt"
    lines = [
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Target distribution:\n{df['target'].value_counts().to_string()}",
        f"\nDescribe:\n{df.describe().to_string()}",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    return output_dir
