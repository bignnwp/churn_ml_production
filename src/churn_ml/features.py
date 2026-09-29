import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the selected feature engineering from the notebook."""
    result = df.copy()

    result["AgeGroup"] = pd.cut(
        result["Age"],
        bins=[0, 25, 35, 50, 100],
        labels=["до 25", "25-35", "35-50", "50+"],
    )

    result["BalanceToSalary"] = result["Balance"] / (result["EstimatedSalary"] + 1e-5)

    result["ActiveMultiProduct"] = (
        (result["IsActiveMember"] == 1) & (result["NumOfProducts"] >= 2)
    ).astype(int)

    result["SegmentFlag"] = (
        (result["Geography"] == "Germany")
        & (result["Gender"] == "Female")
        & (result["IsActiveMember"] == 0)
        & (result["NumOfProducts"] == 1)
    ).astype(int)

    return result
