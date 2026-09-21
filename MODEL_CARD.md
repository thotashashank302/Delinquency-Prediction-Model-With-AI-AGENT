# Model card: Credit-card delinquency predictor

## Model summary

This portfolio project uses a class-weighted
`RandomForestClassifier` to estimate the probability that a credit-card
account will default on its next payment. The model consumes 23 demographic,
credit-limit, repayment-status, bill, and payment-amount features.

The training script saves the estimator and scaler in `models/`, together with
an evaluation report and metadata file generated from the same run.

## Intended use

The model is intended for demonstration, learning, and portfolio evaluation.
It illustrates an end-to-end tabular ML workflow: data loading, preprocessing,
threshold selection, evaluation, artifact creation, and batch scoring.

It must not be used as an automated lending, credit denial, collections, or
customer-contact decision system.

## Evaluation

The default-class decision threshold is `0.35`, selected to prioritize recall
over the default `0.50` threshold. Training reports accuracy, precision, recall,
ROC-AUC, average precision, and a confusion matrix. Run `python -m src.train`
to regenerate the metrics for the checked-in dataset.

Metrics are estimates from one stratified holdout split. They are not a
guarantee of performance on new populations or future time periods.

## Data

The project uses the UCI Default of Credit Card Clients dataset. The source
file is retained under `data/raw/` for reproducible local training. The data
contains sensitive demographic and financial attributes; generated applicant
files are ignored by Git.

## Limitations and risks

- No causal interpretation or explanation is provided for an individual score.
- The dataset is historical and may not represent current customers.
- Fairness, calibration, subgroup performance, drift, and privacy have not
  been established.
- A single random split can produce optimistic or unstable estimates.
- Thresholds should be selected with an explicit business cost matrix.

## Reproducibility

The training script fixes the random seed, uses stratification, records the
feature schema and training configuration, and writes metrics to
`models/evaluation_report.json`. Dependencies are pinned in
`requirements.txt`.
