# Enterprise Payment Exception Risk

## Project overview

This repository is a lightweight research prototype for predicting whether an enterprise payment workflow is likely to require exception review before execution. It uses synthetic data that mimics common payment-operation signals such as invoice amount, approval lag, supplier risk tier, payment method, account-change history, policy flags, duplicate indicators, and scheduling pressure.

The project is intentionally narrow. It is not a production payment system, fraud engine, or broad enterprise automation platform. It is a small reproducible analytics prototype designed to show how payment workflow knowledge can be translated into clean Python code, baseline modeling, improved modeling, evaluation, and documentation.

## Research problem

**Narrow problem:** predict whether a payment request is likely to need manual exception review before payment execution.

The target label is synthetic and represents cases where a payment operation may require additional human review due to rule conflicts, unusual workflow patterns, late approvals, bank-account changes, duplicate signals, or policy exceptions.

## Why the topic matters

Enterprise payment teams often process many routine payment requests while a smaller share require review because of incomplete approvals, mismatched workflow status, unusual timing, or policy constraints. A small risk-scoring prototype can help study how operational signals may support prioritization and auditability without claiming to replace financial controls or human review.

## Repository structure

```text
enterprise-payment-exception-risk/
  README.md
  requirements.txt
  LICENSE
  .gitignore
  src/
    data_generator.py
    preprocessing.py
    baseline.py
    model.py
    evaluation.py
    visualization.py
  experiments/
    run_experiment.py
  docs/
    methodology.md
    limitations.md
    research_positioning.md
  tests/
    test_basic.py
```

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\\Scripts\\activate  # Windows PowerShell
pip install -r requirements.txt
```

## Quick start

Run the full reproducible experiment from the repository root:

```bash
python experiments/run_experiment.py
```

Run tests:

```bash
python -m pytest -q
```

## Methodology summary

The experiment performs the following steps:

1. Generate a synthetic enterprise payment workflow dataset.
2. Split the dataset into training and test sets.
3. Train a baseline logistic regression model using a small feature subset.
4. Train an improved random forest model using richer operational features.
5. Evaluate both models using accuracy, precision, recall, F1, ROC AUC, and average precision.
6. Save metrics and visualizations under `outputs/`.

The baseline model is intentionally simple. The improved model uses additional workflow and control-related features to test whether richer process signals improve risk classification.

## Example output

After running the experiment, the console prints a comparison similar to:

```text
Model comparison
----------------
baseline_logistic_regression: f1=0.63, roc_auc=0.79, average_precision=0.56
improved_random_forest:       f1=0.74, roc_auc=0.88, average_precision=0.70

Artifacts saved to outputs/
```

Exact values may vary slightly depending on package versions, but the random seed is fixed for reproducibility.

## Limitations

This project uses synthetic data and simplified assumptions. It does not represent any real payment platform, customer, institution, or production financial control process. It should not be used to approve, block, or prioritize real payments.

See `docs/limitations.md` for more detail.

## Future work

Possible future extensions include:

- Adding time-series workflow history.
- Testing calibration quality for risk scores.
- Adding explainability reports for reviewer-facing decision support.
- Comparing rule-based thresholds with machine-learning models.
- Evaluating model stability under data drift.

## Disclaimer

This is a research prototype for reproducible technical demonstration only. It is not a production financial control system, compliance tool, or real payment-risk product.
