"""Improved model for payment exception risk."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


def build_improved_model(feature_columns, training_frame):
    """Build a random forest model with richer workflow features."""
    preprocessor = build_preprocessor(feature_columns, training_frame)
    classifier = RandomForestClassifier(
        n_estimators=180,
        max_depth=7,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
