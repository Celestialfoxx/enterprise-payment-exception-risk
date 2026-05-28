# Limitations

This repository is a research prototype and has several important limitations.

## Synthetic data only

The dataset is synthetic. It is designed to resemble workflow-like patterns but does not come from any real enterprise payment platform, supplier database, bank, customer, or ERP system.

## Simplified target label

The target label is generated from a controlled formula. Real exception review decisions depend on organizational policies, approval context, compliance controls, bank validation, payment timing, contractual details, and human judgment.

## Not a production control system

This project should not be used to approve, reject, delay, or prioritize real payments. It does not implement production controls, audit workflows, access control, compliance review, or secure payment operations.

## Limited model scope

The project compares one baseline model and one improved model. It does not include calibration analysis, monitoring, drift detection, explainability dashboards, fairness analysis, or integration with enterprise workflow systems.

## Future work

Potential extensions include:

- adding time-series payment workflow history
- modeling review workload under different thresholds
- testing risk score calibration
- adding feature importance and explanation reports
- simulating data drift across business units
- comparing machine-learning models with deterministic policy rules
