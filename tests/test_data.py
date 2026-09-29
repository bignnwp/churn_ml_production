import pandas as pd

from churn_ml.data import clean_columns


def test_clean_columns_removes_identifiers() -> None:
    df = pd.DataFrame(
        {
            "id": [1],
            "CustomerId": [123],
            "Surname": ["Test"],
            "Age": [40],
        }
    )

    result = clean_columns(df)

    assert list(result.columns) == ["Age"]
