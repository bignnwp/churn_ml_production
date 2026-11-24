"""Тесты модуля мониторинга."""

import numpy as np
import pandas as pd

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


def test_simulate_drift_shifts_lowercase_and_uppercase_age():
    df = pd.DataFrame({"age": [20, 30], "Age": [20, 30], "target": [0, 1]})
    out = simulate_drift(df, age_shift=5.0, target_col=None)
    assert out["age"].tolist() == [25, 35]
    assert out["Age"].tolist() == [25, 35]


def test_compute_label_distribution_shift():
    from churn_ml.monitoring import compute_label_distribution_shift

    reference = pd.DataFrame({"target": [0, 0, 0, 1]})
    current = pd.DataFrame({"target": [0, 1, 1, 1]})
    result = compute_label_distribution_shift(reference, current, "target")

    assert result["reference_positive_rate"] == 0.25
    assert result["current_positive_rate"] == 0.75
    assert result["positive_rate_delta"] == 0.5


def test_extract_drift_summary_reads_nested_evidently_metrics():
    from churn_ml.monitoring import extract_drift_summary

    snapshot = {
        "metrics": [
            {
                "metric": "DriftedColumnsCount",
                "result": {
                    "dataset_drift": True,
                    "share_of_drifted_columns": 0.4,
                    "number_of_drifted_columns": 2,
                    "number_of_columns": 5,
                },
            }
        ]
    }
    result = extract_drift_summary(snapshot)

    assert result["dataset_drift"] is True
    assert result["share_of_drifted_columns"] == 0.4
    assert result["number_of_drifted_columns"] == 2


def test_fairness_report_aligns_non_default_indices():
    y_true = pd.Series([1, 0, 1, 0], index=[10, 20, 30, 40])
    y_pred = np.array([1, 0, 0, 1])
    sensitive = pd.Series(["male", "male", "female", "female"], index=[10, 20, 30, 40])

    result = compute_fairness_report(y_true, y_pred, sensitive, sensitive_name="sex")

    assert set(result["demographic_parity"]["rates"]) == {"male", "female"}
    assert set(result["equalized_odds"]) == {"male", "female"}


def test_prometheus_metrics_are_populated():
    from prometheus_client import generate_latest

    from churn_ml.monitoring import (
        push_drift_metric,
        push_fairness_di,
        push_label_distribution_metrics,
        push_quality_metrics,
    )

    push_quality_metrics({"f1": 0.73})
    push_drift_metric(0.3, alert=True)
    push_fairness_di(0.82)
    push_label_distribution_metrics(0.4, 0.08)
    exposition = generate_latest().decode("utf-8")

    assert "model_f1_score 0.73" in exposition
    assert "model_drift_share 0.3" in exposition
    assert "model_fairness_di 0.82" in exposition
    assert "model_label_positive_rate_delta 0.08" in exposition
