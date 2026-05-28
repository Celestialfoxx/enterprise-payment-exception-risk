"""Visualization helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_comparison(metrics_frame: pd.DataFrame, output_path: str | Path) -> None:
    """Create a bar chart comparing core model metrics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected = metrics_frame[["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]]
    ax = selected.T.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Payment Exception Risk Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Model", loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_risk_distribution(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot invoice amount distribution by target class for exploratory context."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ax = df.boxplot(column="invoice_amount", by="exception_review", figsize=(7, 5))
    ax.set_title("Invoice Amount by Exception Review Label")
    ax.set_xlabel("Exception review required")
    ax.set_ylabel("Invoice amount")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
