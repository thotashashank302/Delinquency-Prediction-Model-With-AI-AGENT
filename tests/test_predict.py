from unittest.mock import Mock

import numpy as np
import pytest

from src import predict


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.29, "Low Risk"),
        (0.30, "Moderate Risk"),
        (0.34, "Moderate Risk"),
        (0.35, "Moderate Risk"),
        (0.66, "Moderate Risk"),
        (0.699999, "Moderate Risk"),
        (0.70, "High Risk"),
    ],
)
def test_single_customer_reports_risk_without_credit_decision(
    probability, expected, monkeypatch, capsys
):
    model, scaler = Mock(), Mock()
    model.predict_proba.return_value = np.array([[1 - probability, probability]])
    monkeypatch.setattr(predict.joblib, "load", Mock(side_effect=[model, scaler]))

    predict.main()

    output = capsys.readouterr().out
    assert f"Default Probability Calculated: {probability * 100:.2f}%" in output
    assert f"Assessed Risk Level: {expected}" in output
    assert "APPROVED" not in output
    assert "DENIED" not in output
    assert "Operational Risk Threshold" not in output
