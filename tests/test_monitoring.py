"""Тесты модуля мониторинга."""

import numpy as np
import pandas as pd
import pytest

from churn_ml.monitoring import (
    compute_fairness_report,
    compute_quality_metrics,
    demographic_parity_difference,
    simulate_drift,
)


def test_simulate_drift_age_shift():
    df = pd.DataFrame({"Age": [20, 30, 40], "target": [0, 1, 0]})
    out = simulate_drift(df, age_shift=5.0, target_col=None)
    assert list(out["Age"]) == [25, 35, 45]


def test_simulate_drift_label_flip():
    df = pd.DataFrame({"Age": np.arange(100), "target": [0] * 100})
    out = simulate_drift(df, age_shift=0, target_col="target", target_flip_rate=0.2, random_state=0)
    assert out["target"].sum() > 0
    assert out["target"].sum() <= 25


def test_demographic_parity():
    y_pred = pd.Series([1, 1, 0, 0, 1, 0])
    sensitive = pd.Series(["a", "a", "a", "b", "b", "b"])
    res = demographic_parity_difference(y_pred, y_pred, sensitive)
    assert "rates" in res
    assert "di" in res


def test_fairness_report():
    y_true = pd.Series([1, 0, 1, 0, 1, 0])
    y_pred = pd.Series([1, 0, 0, 0, 1, 1])
    sensitive = pd.Series(["male", "male", "male", "female", "female", "female"])
    report = compute_fairness_report(y_true, y_pred, sensitive, sensitive_name="sex")
    assert report["sensitive_attribute"] == "sex"
    assert "demographic_parity" in report
    assert "equalized_odds" in report


def test_quality_metrics():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_proba = [0.1, 0.9, 0.4, 0.2]
    m = compute_quality_metrics(y_true, y_pred, y_proba)
    assert "f1" in m
    assert "roc_auc" in m
    assert 0 <= m["f1"] <= 1
