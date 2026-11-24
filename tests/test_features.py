import pandas as pd

from churn_ml.features import add_features


def test_feature_engineering_creates_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "Age": [30, 60],
            "Balance": [100.0, 200.0],
            "EstimatedSalary": [1000.0, 2000.0],
            "IsActiveMember": [1, 0],
            "NumOfProducts": [2, 1],
            "Geography": ["France", "Germany"],
            "Gender": ["Male", "Female"],
        }
    )

    result = add_features(df)

    assert "AgeGroup" in result
    assert "BalanceToSalary" in result
    assert "ActiveMultiProduct" in result
    assert "SegmentFlag" in result
    assert result.loc[0, "ActiveMultiProduct"] == 1
    assert result.loc[1, "SegmentFlag"] == 1
