"""Tests for dataset-specific transformations in the practical monitoring task."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "practical_assignment.py"
SPEC = spec_from_file_location("practical_assignment", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load scripts/practical_assignment.py")
practical_assignment = module_from_spec(SPEC)
SPEC.loader.exec_module(practical_assignment)


def test_credit_g_personal_status_codes_are_mapped_to_sex(tmp_path, monkeypatch):
    dataset = pd.DataFrame(
        {
            "class": ["good", "bad", "good", "bad", "good"],
            "personal_status": ["A91", "A92", "A93", "A94", "A95"],
            "age": [22, 31, 45, 50, 28],
            "duration": [6, 12, 18, 24, 9],
        }
    )
    cache = tmp_path / "credit_g_openml.csv"
    dataset.to_csv(cache, index=False)
    monkeypatch.setattr(practical_assignment, "DATA_CACHE", cache)

    frame, target_col, sensitive_col = practical_assignment.step01_load_dataset()

    assert target_col == "class"
    assert sensitive_col == "sex"
    assert frame["sex"].tolist() == ["male", "female", "male", "male", "female"]
    assert frame[target_col].tolist() == [0, 1, 0, 1, 0]


def test_step03_simulate_drift_changes_age_and_target_share():
    frame = pd.DataFrame(
        {
            "age": list(range(100)),
            "class": [0] * 70 + [1] * 30,
        }
    )

    drifted = practical_assignment.step03_simulate_drift(frame, "class")

    assert (drifted["age"] - frame["age"]).eq(5).all()
    assert drifted["class"].mean() != frame["class"].mean()
    assert frame["age"].equals(pd.Series(range(100), name="age"))


def test_training_pipeline_excludes_sensitive_columns_and_computes_fairness(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        {
            "age": np.arange(240) % 70 + 18,
            "duration": np.arange(240) % 48 + 1,
            "checking_status": ["A11", "A12", "A13", "A14"] * 60,
            "personal_status": ["A91", "A92", "A93", "A94"] * 60,
            "sex": ["male", "female", "male", "male"] * 60,
            "class": [0, 1] * 120,
        }
    )
    monkeypatch.setattr(practical_assignment, "REPORTS_DIR", tmp_path)

    model, _, _, X_test, y_test, feature_cols = practical_assignment.step02_train_model(
        frame, "class"
    )
    sensitive = frame.loc[X_test.index, "sex"]
    report = practical_assignment.step05_fairness(model, X_test, y_test, sensitive)

    assert "sex" not in feature_cols
    assert "personal_status" not in feature_cols
    assert "demographic_parity" in report
    assert (tmp_path / "fairness_report.json").is_file()
