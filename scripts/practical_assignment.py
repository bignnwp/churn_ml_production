#!/usr/bin/env python3
"""End-to-end practical assignment: ML model monitoring.

Run locally from the repository root:
    poetry run python scripts/practical_assignment.py --hold-seconds 60

Run as a long-lived Prometheus exporter:
    poetry run python scripts/practical_assignment.py --metrics-port 8001 --serve-forever

The script downloads the OpenML ``credit-g`` dataset on first run and caches it
under ``data/raw/``. Generated reports are written to ``reports/practical_assignment/``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from threading import Event
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from churn_ml.monitoring import (
    build_evidently_drift_report,
    compute_fairness_report,
    compute_label_distribution_shift,
    compute_quality_metrics,
    extract_drift_summary,
    push_drift_metric,
    push_fairness_di,
    push_label_distribution_metrics,
    push_quality_metrics,
    simulate_drift,
    start_metrics_server,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_CACHE = ROOT / "data" / "raw" / "credit_g_openml.csv"
REPORTS_DIR = ROOT / "reports" / "practical_assignment"
CREDIT_G_STATUS_SEX = {
    "a91": "male",
    "a93": "male",
    "a94": "male",
    "a92": "female",
    "a95": "female",
}


def _map_credit_g_sex(personal_status: str) -> str | None:
    """Map German Credit personal-status codes or readable labels to sex groups."""
    value = personal_status.strip().lower()
    if value in CREDIT_G_STATUS_SEX:
        return CREDIT_G_STATUS_SEX[value]
    if value.startswith("male"):
        return "male"
    if value.startswith("female"):
        return "female"
    return None


def step01_load_dataset() -> tuple[pd.DataFrame, str, str]:
    """01 — Load German Credit (OpenML credit-g) and derive the sex group."""
    print("\n" + "=" * 68)
    print("01  Загрузка датасета OpenML credit-g")
    print("=" * 68)

    if DATA_CACHE.exists():
        df = pd.read_csv(DATA_CACHE)
        print(f"  Используется локальный кэш: {DATA_CACHE}")
    else:
        from sklearn.datasets import fetch_openml

        try:
            dataset = fetch_openml(
                name="credit-g",
                version=1,
                as_frame=True,
                parser="auto",
            )
        except Exception as exc:
            raise RuntimeError(
                "Не удалось загрузить OpenML credit-g. Проверьте доступ к интернету "
                "или положите CSV-датасет в data/raw/credit_g_openml.csv."
            ) from exc

        df = dataset.frame.copy()
        DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_CACHE, index=False)
        print(f"  Датасет закэширован: {DATA_CACHE}")

    target_col = "class"
    sensitive_col = "sex"
    if target_col not in df.columns:
        raise ValueError(f"В credit-g отсутствует целевая колонка {target_col!r}")
    if "personal_status" not in df.columns:
        raise ValueError("В credit-g отсутствует колонка personal_status для fairness по sex")
    if "age" not in df.columns:
        raise ValueError("В credit-g отсутствует колонка age для симуляции data drift")

    df[target_col] = (df[target_col].astype(str).str.lower() == "bad").astype(int)
    status = df["personal_status"].astype(str)
    # OpenML credit-g commonly uses A91-A95 codes; readable labels are supported too.
    derived_sex = status.map(_map_credit_g_sex)
    if derived_sex.isna().any():
        unknown = sorted(status[derived_sex.isna()].unique().tolist())
        raise ValueError(f"Не удалось сопоставить personal_status с sex: {unknown}")
    df[sensitive_col] = derived_sex

    if df.empty:
        raise ValueError("OpenML credit-g вернул пустой датасет")

    print(f"  Строк: {len(df)}, признаков с target: {df.shape[1]}")
    print(f"  Доля положительного класса (bad): {df[target_col].mean():.3f}")
    print(f"  Группы sex: {df[sensitive_col].value_counts().to_dict()}")
    print(f"  Age: min={df['age'].min()}, max={df['age'].max()}, " f"mean={df['age'].mean():.1f}")
    return df, target_col, sensitive_col


def step02_train_model(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[Any, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """02 — Train CatBoost if installed, otherwise a sklearn RandomForest pipeline."""
    print("\n" + "=" * 68)
    print("02  Обучение модели")
    print("=" * 68)

    excluded = {target_col, "sex", "personal_status"}
    feature_cols = [column for column in df.columns if column not in excluded]
    X = df[feature_cols].copy()
    y = df[target_col].astype(int).copy()
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        CatBoostClassifier = None

    if CatBoostClassifier is not None:
        print("  Модель: CatBoostClassifier")
        model = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            loss_function="Logloss",
            eval_metric="F1",
            random_seed=42,
            verbose=False,
            cat_features=cat_cols,
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=30)
        estimator: Any = model
    else:
        print("  CatBoost не установлен — используется RandomForest + sklearn Pipeline")
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", RobustScaler()),
                        ]
                    ),
                    num_cols,
                ),
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    cat_cols,
                ),
            ],
            remainder="drop",
        )
        estimator = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)[:, 1]
    metrics = compute_quality_metrics(y_test, y_pred, y_proba)
    print(f"  Test metrics: {json.dumps(metrics, indent=2, ensure_ascii=False)}")
    print(classification_report(y_test, y_pred, target_names=["good", "bad"], zero_division=0))
    return estimator, X_train, y_train, X_test, y_test, feature_cols


def step03_simulate_drift(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """03 — Shift age by +5 years and flip 15% of binary target labels."""
    print("\n" + "=" * 68)
    print("03  Симуляция covariate drift и label shift")
    print("=" * 68)

    drifted = simulate_drift(
        df,
        age_shift=5.0,
        target_col=target_col,
        target_flip_rate=0.15,
        random_state=42,
    )
    age_before = float(df["age"].mean())
    age_after = float(drifted["age"].mean())
    label_before = float(df[target_col].mean())
    label_after = float(drifted[target_col].mean())
    print(f"  Age mean: {age_before:.2f} → {age_after:.2f} (+{age_after - age_before:.2f})")
    print(f"  Positive class share: {label_before:.3f} → {label_after:.3f}")
    return drifted


def step04_evidently_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> dict[str, Any]:
    """04 — Run Evidently drift report over features and the target distribution."""
    print("\n" + "=" * 68)
    print("04  Evidently: поиск data drift и label shift")
    print("=" * 68)

    columns = list(dict.fromkeys([*feature_cols, target_col]))
    ref = reference[columns].copy()
    cur = current[columns].copy()
    # Treat binary target as categorical so Evidently tests class proportions.
    ref[target_col] = ref[target_col].astype(str).astype("category")
    cur[target_col] = cur[target_col].astype(str).astype("category")

    html_path = REPORTS_DIR / "evidently_data_drift.html"
    snapshot = build_evidently_drift_report(ref, cur, output_html=html_path)
    summary = extract_drift_summary(snapshot)
    label_shift = compute_label_distribution_shift(reference, current, target_col)
    label_drift = abs(label_shift["positive_rate_delta"]) >= 0.05

    summary["label_distribution"] = label_shift
    summary["label_distribution_drift_detected"] = label_drift
    summary["dataset_drift"] = bool(summary.get("dataset_drift", False) or label_drift)
    if summary.get("share_of_drifted_columns") is None:
        # Evidently 0.7 reports may expose per-column metrics instead of the
        # legacy aggregate fields. Preserve a conservative label-drift signal.
        summary["share_of_drifted_columns"] = (1.0 / len(columns)) if label_drift else 0.0

    summary_path = REPORTS_DIR / "evidently_data_drift.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"  HTML report: {html_path}")
    print(f"  JSON summary: {summary_path}")
    print(f"  Evidently dataset drift: {summary.get('dataset_drift')}")
    print(f"  Share of drifted columns: {summary.get('share_of_drifted_columns')}")
    print(
        "  Positive class share: "
        f"{label_shift['reference_positive_rate']:.3f} → "
        f"{label_shift['current_positive_rate']:.3f} "
        f"(delta={label_shift['positive_rate_delta']:+.3f})"
    )
    return summary


def step05_fairness(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive: pd.Series,
) -> dict[str, Any]:
    """05 — Compute demographic parity and equalized odds by derived sex group."""
    print("\n" + "=" * 68)
    print("05  Fairness по sex")
    print("=" * 68)

    y_pred = model.predict(X_test)
    report = compute_fairness_report(y_test, y_pred, sensitive, sensitive_name="sex")
    demographic_parity = report["demographic_parity"]
    print(f"  Positive prediction rates: {demographic_parity['rates']}")
    print(f"  Demographic Parity Difference: {demographic_parity['dpd']:.4f}")
    print(f"  Disparate Impact ratio: {demographic_parity['di']:.4f}")
    print(f"  Equalized Odds: {json.dumps(report['equalized_odds'], ensure_ascii=False)}")
    print(f"  Ограничение интерпретации: {report['note']}")

    out_path = REPORTS_DIR / "fairness_report.json"
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"  JSON report: {out_path}")
    return report


def step06_prometheus(
    metrics: dict[str, float],
    drift_summary: dict[str, Any],
    fairness_report: dict[str, Any],
    port: int,
    hold_seconds: int,
    serve_forever: bool,
) -> None:
    """06 — Expose F1, drift, label distribution and fairness metrics to Prometheus."""
    print("\n" + "=" * 68)
    print("06  Prometheus: экспорт метрик и alert rules")
    print("=" * 68)

    start_metrics_server(port)
    push_quality_metrics(metrics)

    drift_share = float(drift_summary.get("share_of_drifted_columns") or 0.0)
    drift_alert = bool(drift_summary.get("dataset_drift")) or drift_share > 0.25
    push_drift_metric(drift_share, alert=drift_alert)

    label_distribution = drift_summary.get("label_distribution", {})
    if label_distribution:
        push_label_distribution_metrics(
            float(label_distribution["current_positive_rate"]),
            float(label_distribution["positive_rate_delta"]),
        )

    di = fairness_report.get("demographic_parity", {}).get("di")
    if di is not None:
        push_fairness_di(float(di))

    print(f"  Exporter: http://127.0.0.1:{port}/metrics")
    print(f"  model_f1_score: {metrics.get('f1', float('nan')):.4f}")
    print(f"  model_drift_share: {drift_share:.4f}; drift alert condition: {drift_alert}")
    print(f"  model_label_positive_rate_delta: {label_distribution.get('positive_rate_delta')}")
    print(f"  model_fairness_di: {di}")
    print("  Prometheus rules: monitoring/prometheus_rules.yml")

    if serve_forever:
        print("  Exporter работает до остановки контейнера (Ctrl+C).")
        try:
            Event().wait()
        except KeyboardInterrupt:
            print("  Exporter остановлен пользователем.")
    elif hold_seconds > 0:
        print(f"  Exporter будет доступен ещё {hold_seconds} сек.")
        try:
            time.sleep(hold_seconds)
        except KeyboardInterrupt:
            print("  Остановлено пользователем.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Практическое задание: ML monitoring")
    parser.add_argument("--metrics-port", type=int, default=8001, help="Порт Prometheus exporter")
    parser.add_argument(
        "--hold-seconds",
        type=int,
        default=60,
        help="Сколько секунд держать exporter (0 = завершить сразу)",
    )
    parser.add_argument(
        "--serve-forever",
        action="store_true",
        help="Не завершать процесс exporter (для Docker Compose/Prometheus)",
    )
    args = parser.parse_args()
    if not 1 <= args.metrics_port <= 65535:
        parser.error("--metrics-port должен быть в диапазоне 1..65535")
    if args.hold_seconds < 0:
        parser.error("--hold-seconds не может быть отрицательным")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Практическое задание: система мониторинга ML-модели")
    print(f"Отчёты: {REPORTS_DIR}")

    df, target_col, sensitive_col = step01_load_dataset()
    model, _, _, X_test, y_test, feature_cols = step02_train_model(df, target_col)
    drifted_df = step03_simulate_drift(df, target_col)
    drift_summary = step04_evidently_drift(df, drifted_df, feature_cols, target_col)

    sensitive_test = df.loc[X_test.index, sensitive_col]
    fairness_report = step05_fairness(model, X_test, y_test, sensitive_test)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    quality_metrics = compute_quality_metrics(y_test, y_pred, y_proba)

    summary = {
        "dataset": "OpenML credit-g",
        "model": type(model).__name__,
        "quality_metrics": quality_metrics,
        "drift_summary": drift_summary,
        "fairness": fairness_report,
    }
    summary_path = REPORTS_DIR / "assignment_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nСводный отчёт: {summary_path}")

    step06_prometheus(
        quality_metrics,
        drift_summary,
        fairness_report,
        port=args.metrics_port,
        hold_seconds=args.hold_seconds,
        serve_forever=args.serve_forever,
    )
    print("\nВсе 6 шагов практического задания выполнены.")


if __name__ == "__main__":
    main()
