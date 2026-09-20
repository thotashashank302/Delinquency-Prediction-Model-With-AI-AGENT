 # Credit Card Delinquency Prediction

This project trains a scikit-learn random-forest model to estimate the
probability that a credit-card account will default next month. It provides
training, single-customer, and opt-in batch reporting workflows.

## Structure

- `src/common.py` — canonical features, paths, risk labels, and validation.
- `src/train.py` — trains and saves the model and scaler.
- `src/predict.py` — assesses the built-in example customer.
- `src/batch_predict.py` — validates a CSV, creates a report, and optionally
  emails moderate-risk reminders.
- `data/raw/` — committed source dataset; `models/` — model artifacts.

## Setup

Use Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training and prediction

These commands work from any directory:

```bash
python -m src.train
python -m src.predict
python -m src.batch_predict
```

Training expects `data/raw/default of credit card clients.xls` and writes
`models/random_forest_model.pkl` and `models/scaler.pkl`. Paths are resolved
relative to the repository, not the current working directory.

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

## Testing

```bash
pytest -q
python -m compileall src
```

Tests use in-memory model stubs and do not connect to SMTP.

## Limitations and data/model notes

This is a demonstration model, not financial or credit-underwriting advice.
Predictions depend on the historical UCI credit-card dataset and saved
artifacts; data drift, fairness, calibration, and regulatory requirements are
not addressed. Retrain and validate before production use.