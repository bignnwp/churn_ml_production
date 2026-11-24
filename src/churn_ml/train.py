import argparse
from pathlib import Path

import joblib

from .config import TrainConfig, ensure_parent
from .data import clean_columns, load_data
from .features import add_features
from .modeling import train_random_forest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the churn Random Forest model.")
    parser.add_argument("--data", required=True, help="Path to raw train.csv.")
    parser.add_argument(
        "--model-out",
        default="models/churn_random_forest.joblib",
        help="Path for the trained model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig()

    df = load_data(args.data)
    df = clean_columns(df)
    df = add_features(df)

    X = df.drop(columns=[config.target])
    y = df[config.target]

    model, metrics = train_random_forest(X, y, config)

    model_path = Path(args.model_out)
    ensure_parent(model_path)
    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
