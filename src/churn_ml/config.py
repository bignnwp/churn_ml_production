from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    target: str = "Exited"
    test_size: float = 0.2
    random_state: int = 42
    cv_splits: int = 3
    scoring: str = "recall"
    n_jobs: int = -1


NUMERIC_FEATURES = [
    "Age",
    "NumOfProducts",
    "Balance",
    "BalanceToSalary",
    "ActiveMultiProduct",
    "SegmentFlag",
]

CATEGORICAL_FEATURES = [
    "IsActiveMember",
    "Geography",
    "Gender",
    "AgeGroup",
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
