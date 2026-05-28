"""Synthetic data generation for enterprise payment exception risk.

The generated data is not based on any real company, customer, or payment
system. It is designed to support a small reproducible modeling experiment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Return the logistic transform of an array."""
    return 1 / (1 + np.exp(-x))


def generate_payment_data(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic enterprise payment workflow dataset.

    Parameters
    ----------
    n_samples:
        Number of payment requests to generate.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Synthetic payment workflow records with a binary target column named
        ``exception_review``.
    """
    rng = np.random.default_rng(random_state)

    supplier_risk_tier = rng.choice(["low", "medium", "high"], size=n_samples, p=[0.58, 0.30, 0.12])
    payment_method = rng.choice(["ach", "wire", "check", "virtual_card"], size=n_samples, p=[0.55, 0.22, 0.13, 0.10])
    business_unit = rng.choice(["operations", "sales", "finance", "it", "legal"], size=n_samples)

    invoice_amount = rng.lognormal(mean=8.15, sigma=0.85, size=n_samples).round(2)
    approval_lag_days = rng.gamma(shape=2.1, scale=2.5, size=n_samples).round(1)
    days_until_due = rng.normal(loc=9, scale=7, size=n_samples).round(1)
    prior_exceptions_90d = rng.poisson(lam=0.35, size=n_samples)
    account_changes_180d = rng.binomial(1, p=0.08, size=n_samples)
    duplicate_invoice_signal = rng.binomial(1, p=0.07, size=n_samples)
    missing_supporting_document = rng.binomial(1, p=0.10, size=n_samples)
    policy_flag_count = rng.poisson(lam=0.55, size=n_samples)
    approval_chain_length = rng.integers(1, 6, size=n_samples)
    schedule_pressure = (days_until_due < 2).astype(int)
    amount_z_proxy = (np.log1p(invoice_amount) - np.mean(np.log1p(invoice_amount))) / np.std(np.log1p(invoice_amount))

    risk_tier_score = pd.Series(supplier_risk_tier).map({"low": 0.0, "medium": 0.65, "high": 1.3}).to_numpy()
    wire_score = (payment_method == "wire").astype(float) * 0.6

    latent_risk = (
        -2.4
        + 0.55 * amount_z_proxy
        + 0.33 * approval_lag_days / 3
        + 0.70 * risk_tier_score
        + 0.75 * account_changes_180d
        + 1.15 * duplicate_invoice_signal
        + 0.82 * missing_supporting_document
        + 0.55 * policy_flag_count
        + 0.35 * schedule_pressure
        + 0.20 * approval_chain_length
        + wire_score
        + rng.normal(0, 0.75, n_samples)
    )

    probability = sigmoid(latent_risk)
    exception_review = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "payment_request_id": [f"PAY-{i:05d}" for i in range(n_samples)],
            "supplier_risk_tier": supplier_risk_tier,
            "payment_method": payment_method,
            "business_unit": business_unit,
            "invoice_amount": invoice_amount,
            "approval_lag_days": approval_lag_days,
            "days_until_due": days_until_due,
            "prior_exceptions_90d": prior_exceptions_90d,
            "account_changes_180d": account_changes_180d,
            "duplicate_invoice_signal": duplicate_invoice_signal,
            "missing_supporting_document": missing_supporting_document,
            "policy_flag_count": policy_flag_count,
            "approval_chain_length": approval_chain_length,
            "schedule_pressure": schedule_pressure,
            "exception_review": exception_review,
        }
    )


if __name__ == "__main__":
    print(generate_payment_data().head())
