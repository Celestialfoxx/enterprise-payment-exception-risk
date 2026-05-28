"""Baseline model for payment exception risk."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


def build_baseline_model(feature_columns, training_frame):
    """Build a simple logistic regression baseline.

    The baseline uses a small feature subset to create a transparent reference
    point for comparison with the improved model.
    """
    preprocessor = build_preprocessor(feature_columns, training_frame)
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
