from pathlib import Path

import pandas as pd

DROP_COLUMNS = ["id", "CustomerId", "Surname"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw churn dataset from a local file."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifiers that were excluded in the original project."""
    return df.drop(columns=DROP_COLUMNS, errors="ignore").copy()
