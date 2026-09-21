# Credit Card Delinquency Prediction

[![Quality checks](https://github.com/thotashashank302/Delinquency-Prediction-Model-With-AI-AGENT/actions/workflows/ci.yml/badge.svg)](https://github.com/thotashashank302/Delinquency-Prediction-Model-With-AI-AGENT/actions/workflows/ci.yml)

A portfolio-grade, end-to-end tabular machine-learning project that estimates
the probability of next-month credit-card payment default. It demonstrates
reproducible training, explicit risk-threshold selection, model evaluation,
batch scoring, opt-in email notifications, tests, and automated CI.

> **Portfolio scope:** This is a demonstration project, not a production
> lending or credit-underwriting system. See [MODEL_CARD.md](MODEL_CARD.md) for
> intended use, evaluation, data limitations, and risks.

## Structure

- `src/common.py` — canonical features, paths, risk labels, and validation.
- `src/train.py` — trains and saves the model and scaler.
- `src/predict.py` — assesses the built-in example customer.
- `src/batch_predict.py` — validates a CSV, creates a report, and optionally
  emails moderate-risk reminders.
- `data/raw/` — committed source dataset; `models/` — model artifacts.
- `tests/` — deterministic unit tests with no network calls.
- `.github/workflows/ci.yml` — lint, compile, and test checks.
- `MODEL_CARD.md` — model scope, evaluation, and limitations.

## Setup

Use Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Training and prediction

These commands work from any directory:

```bash
python -m src.train
python -m src.predict
python -m src.batch_predict
```

Training expects `data/raw/default of credit card clients.xls` and writes the
model, scaler, evaluation metrics, and run metadata to `models/`. Paths are
resolved relative to the repository, not the current working directory:

```text
models/random_forest_model.pkl
models/scaler.pkl
models/evaluation_report.json
models/model_metadata.json
```

The training run uses a stratified holdout split, a fixed random seed, class
balancing, and a `0.35` probability threshold chosen to emphasize recall.

## Batch input schema

Batch CSVs must contain the 23 numeric columns in
`src.common.FEATURE_NAMES` and an `Email` column. The output adds
`Delinquency_Probability_%` and `Assessed_Risk_Level` (`Low Risk`,
`Moderate Risk`, or `High Risk`). Generated batch files are ignored by Git.

## Email security

The batch command asks interactively for an address and hidden app password.
SMTP uses STARTTLS, and each reminder requires explicit confirmation. Use a
provider-issued app password, never a primary account password, and do not put
credentials in source control.

## Quality checks

```bash
pytest
ruff check src tests
python -m compileall src
```

The test suite uses in-memory model stubs and never connects to SMTP. CI runs
the same checks on every push and pull request.

## Limitations and data/model notes

Predictions depend on the historical UCI credit-card dataset and saved
artifacts. Model performance is based on one holdout split; fairness,
calibration, drift, explainability, and regulatory requirements are outside
this portfolio scope. Do not use the output as an automated financial
decision.