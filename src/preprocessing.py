"""Preprocessing helpers for payment exception risk modeling."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "exception_review"
ID_COLUMNS = ["payment_request_id"]

BASELINE_FEATURES = [
    "invoice_amount",
    "approval_lag_days",
    "supplier_risk_tier",
]

IMPROVED_FEATURES = [
    "supplier_risk_tier",
    "payment_method",
    "business_unit",
    "invoice_amount",
    "approval_lag_days",
    "days_until_due",
    "prior_exceptions_90d",
    "account_changes_180d",
    "duplicate_invoice_signal",
    "missing_supporting_document",
    "policy_flag_count",
    "approval_chain_length",
    "schedule_pressure",
]


def split_features_target(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and target vector."""
    missing = set(feature_columns + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df[feature_columns].copy(), df[TARGET_COLUMN].copy()


def build_preprocessor(feature_columns: List[str], df: pd.DataFrame) -> ColumnTransformer:
    """Build a column transformer for numeric and categorical features."""
    categorical_features = [col for col in feature_columns if df[col].dtype == "object"]
    numeric_features = [col for col in feature_columns if col not in categorical_features]

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )
