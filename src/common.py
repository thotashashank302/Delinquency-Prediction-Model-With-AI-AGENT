"""Shared configuration and validation for the delinquency model."""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "random_forest_model.pkl"
SCALER_PATH = REPO_ROOT / "models" / "scaler.pkl"
METADATA_PATH = REPO_ROOT / "models" / "model_metadata.json"
EVALUATION_PATH = REPO_ROOT / "models" / "evaluation_report.json"
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "default of credit card clients.xls"
DEFAULT_RISK_THRESHOLD = 0.35
RANDOM_STATE = 42

FEATURE_NAMES = (
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
)


def classify_risk(probability: float) -> str:
    """Return the report label for a default probability in the [0, 1] range."""
    if not 0 <= probability <= 1:
        raise ValueError(f"Probability must be between 0 and 1; received {probability!r}.")
    if probability < 0.30:
        return "Low Risk"
    if probability < 0.70:
        return "Moderate Risk"
    return "High Risk"


def validate_feature_columns(frame: pd.DataFrame, *, require_email: bool = False) -> None:
    """Validate required columns and ensure model inputs are numeric."""
    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if require_email and "Email" not in frame.columns:
        missing.append("Email")
    if missing:
        raise ValueError("Input data is missing required column(s): " + ", ".join(missing))

    non_numeric = [name for name in FEATURE_NAMES if not pd.api.types.is_numeric_dtype(frame[name])]
    if non_numeric:
        raise ValueError("Feature column(s) must contain numeric values: " + ", ".join(non_numeric))
    if frame.loc[:, FEATURE_NAMES].isna().any().any():
        missing_values = [name for name in FEATURE_NAMES if frame[name].isna().any()]
        raise ValueError("Feature column(s) contain missing values: " + ", ".join(missing_values))
