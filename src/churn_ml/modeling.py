from typing import Any

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TrainConfig


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(config: TrainConfig, use_smote: bool = False) -> ImbPipeline:
    steps: list[tuple[str, Any]] = [("preprocessor", build_preprocessor())]

    if use_smote:
        steps.append(("smote", SMOTE(random_state=config.random_state)))

    steps.append(
        (
            "clf",
            RandomForestClassifier(
                random_state=config.random_state,
                n_jobs=config.n_jobs,
            ),
        )
    )
    return ImbPipeline(steps=steps)


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainConfig,
) -> tuple[Any, dict[str, Any]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )

    cv = StratifiedKFold(
        n_splits=config.cv_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    pipeline = build_pipeline(config, use_smote=False)
    param_dist = {
        "clf__n_estimators": randint(100, 500),
        "clf__max_depth": randint(4, 30),
        "clf__min_samples_leaf": randint(1, 10),
        "clf__min_samples_split": randint(2, 20),
        "clf__max_features": ["sqrt", 0.3, 0.5, 0.8, None],
        "clf__bootstrap": [True, False],
        "clf__class_weight": ["balanced", "balanced_subsample"],
    }

    search = HalvingRandomSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        cv=cv,
        scoring=config.scoring,
        n_candidates=30,
        factor=3,
        resource="n_samples",
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
        "best_params": search.best_params_,
    }
    return best_model, metrics
