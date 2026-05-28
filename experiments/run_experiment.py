"""Run the reproducible payment exception risk experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline import build_baseline_model
from src.data_generator import generate_payment_data
from src.evaluation import evaluate_binary_classifier, metrics_to_frame
from src.model import build_improved_model
from src.preprocessing import BASELINE_FEATURES, IMPROVED_FEATURES, split_features_target
from src.visualization import plot_metric_comparison, plot_risk_distribution


def run_experiment() -> pd.DataFrame:
    """Generate data, train models, evaluate, and save artifacts."""
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    data = generate_payment_data(n_samples=2500, random_state=42)
    data.to_csv(output_dir / "synthetic_payment_workflows.csv", index=False)

    train_df, test_df = train_test_split(
        data,
        test_size=0.25,
        random_state=42,
        stratify=data["exception_review"],
    )

    X_train_base, y_train = split_features_target(train_df, BASELINE_FEATURES)
    X_test_base, y_test = split_features_target(test_df, BASELINE_FEATURES)
    baseline = build_baseline_model(BASELINE_FEATURES, train_df)
    baseline.fit(X_train_base, y_train)

    X_train_improved, _ = split_features_target(train_df, IMPROVED_FEATURES)
    X_test_improved, _ = split_features_target(test_df, IMPROVED_FEATURES)
    improved = build_improved_model(IMPROVED_FEATURES, train_df)
    improved.fit(X_train_improved, y_train)

    metrics = {
        "baseline_logistic_regression": evaluate_binary_classifier(baseline, X_test_base, y_test),
        "improved_random_forest": evaluate_binary_classifier(improved, X_test_improved, y_test),
    }

    metrics_frame = metrics_to_frame(metrics)
    metrics_frame.to_csv(output_dir / "metrics.csv")
    plot_metric_comparison(metrics_frame, output_dir / "metric_comparison.png")
    plot_risk_distribution(data, output_dir / "amount_distribution.png")

    print("\nModel comparison")
    print("----------------")
    for model_name, row in metrics_frame.iterrows():
        print(
            f"{model_name}: "
            f"f1={row['f1']:.3f}, "
            f"roc_auc={row['roc_auc']:.3f}, "
            f"average_precision={row['average_precision']:.3f}"
        )
    print("\nArtifacts saved to outputs/")
    return metrics_frame


if __name__ == "__main__":
    run_experiment()
