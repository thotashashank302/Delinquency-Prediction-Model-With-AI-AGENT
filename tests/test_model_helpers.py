import numpy as np
import pandas as pd
import pytest

from src.batch_predict import prepare_report
from src.common import FEATURE_NAMES, classify_risk, validate_feature_columns


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.0, "Low Risk"), (0.299999, "Low Risk"), (0.30, "Moderate Risk"),
     (0.699999, "Moderate Risk"), (0.70, "High Risk"), (1.0, "High Risk")],
)
def test_classify_risk_boundaries(probability, expected):
    assert classify_risk(probability) == expected


def test_validate_batch_input_reports_missing_columns():
    with pytest.raises(ValueError, match="missing required column"):
        validate_feature_columns(pd.DataFrame({"Email": ["a@example.com"]}), require_email=True)


def test_prepare_report_validates_and_adds_report_columns():
    frame = pd.DataFrame(
        [[*[1] * len(FEATURE_NAMES), "a@example.com"]],
        columns=[*FEATURE_NAMES, "Email"],
    )

    class FakeScaler:
        def transform(self, values):
            return values

    class FakeModel:
        def predict_proba(self, values):
            return np.array([[0.8, 0.2]])

    report = prepare_report(frame, FakeModel(), FakeScaler())
    assert report.loc[0, "Delinquency_Probability_%"] == 20.0
    assert report.loc[0, "Assessed_Risk_Level"] == "Low Risk"
