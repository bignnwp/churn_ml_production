"""Integration smoke test for the locked Evidently 0.7.x API."""

import pandas as pd
import pytest

from churn_ml.monitoring import build_evidently_drift_report, extract_drift_summary


def test_evidently_builds_html_drift_report(tmp_path):
    pytest.importorskip("evidently")
    reference = pd.DataFrame(
        {
            "age": list(range(40)),
            "balance": [float(value * 10) for value in range(40)],
            "segment": ["A" if value % 2 else "B" for value in range(40)],
        }
    )
    current = reference.copy()
    current["age"] = current["age"] + 5
    current["balance"] = current["balance"] + 500

    html_path = tmp_path / "drift.html"
    snapshot = build_evidently_drift_report(reference, current, output_html=html_path)
    summary = extract_drift_summary(snapshot)

    assert html_path.is_file()
    assert html_path.stat().st_size > 0
    assert "metrics" in summary
