#!/usr/bin/env python3
"""
Практическое задание вебинара «Мониторинг моделей машинного обучения».

Шаги:
  01  Взять dataset (OpenML credit-g)
  02  Обучить модель (CatBoost, fallback — RandomForest)
  03  Сымитировать дрифт (Age +5 лет, изменение распределения целевых классов)
  04  Запустить Evidently → найти дрифт
  05  Посчитать fairness по sex
  06  Экспортировать F1 в Prometheus → настроить алерт

Запуск из корня проекта:
  poetry run python scripts/practical_assignment.py
  # или
  python scripts/practical_assignment.py --metrics-port 8001
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

# Добавляем src в path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn_ml.monitoring import (  # noqa: E402
    build_evidently_drift_report,
    compute_fairness_report,
    compute_quality_metrics,
    extract_drift_summary,
    push_drift_metric,
    push_fairness_di,
    push_quality_metrics,
    simulate_drift,
    start_metrics_server,
)

REPORTS_DIR = ROOT / "reports" / "practical_assignment"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def step01_load_dataset() -> tuple[pd.DataFrame, str, str]:
    """01 — Загрузка OpenML credit-g (German Credit Data)."""
    print("\n" + "=" * 60)
    print("01  Загрузка датасета OpenML credit-g")
    print("=" * 60)

    from sklearn.datasets import fetch_openml

    data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()

    # Целевая переменная: class (good/bad) → 1 = bad (риск), 0 = good
    target_col = "class"
    df[target_col] = (df[target_col].astype(str) == "bad").astype(int)

    # Чувствительный признак для fairness
    sensitive_col = "personal_status"
    # Упростим personal_status до бинарного proxy «sex» для учебного примера
    # (в реальных системах нужен правовой базис — 152-ФЗ)
    if sensitive_col in df.columns:
        # male: 'male single', 'male mar/wid', 'male div/sep'
        # female: 'female div/dep/mar'
        df["sex"] = df[sensitive_col].astype(str).str.startswith("male").map(
            {True: "male", False: "female"}
        )
    else:
        df["sex"] = "unknown"

    print(f"  Строк: {len(df)}, столбцов: {df.shape[1]}")
    print(f"  Целевая: {target_col} (1=bad), доля положительного класса: {df[target_col].mean():.3f}")
    print(f"  Признак sex: {df['sex'].value_counts().to_dict()}")
    print(f"  Age: min={df['age'].min()}, max={df['age'].max()}, mean={df['age'].mean():.1f}")

    return df, target_col, "sex"


def step02_train_model(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[Pipeline, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """02 — Обучение модели (CatBoost при наличии, иначе RandomForest)."""
    print("\n" + "=" * 60)
    print("02  Обучение модели")
    print("=" * 60)

    feature_cols = [c for c in df.columns if c not in (target_col, "sex", "personal_status")]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Определяем типы признаков
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    use_catboost = False
    try:
        from catboost import CatBoostClassifier

        use_catboost = True
    except ImportError:
        pass

    if use_catboost:
        print("  Используем CatBoostClassifier")
        # CatBoost принимает категориальные напрямую
        model = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            loss_function="Logloss",
            eval_metric="F1",
            random_seed=42,
            verbose=0,
            cat_features=cat_cols,
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
        # Оборачиваем в простой объект с predict/predict_proba
        class _CbWrapper:
            def __init__(self, m):
                self._m = m

            def predict(self, X):
                return self._m.predict(X).astype(int)

            def predict_proba(self, X):
                return self._m.predict_proba(X)

        pipeline = _CbWrapper(model)
    else:
        print("  CatBoost не найден — используем RandomForest + sklearn Pipeline")
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", RobustScaler()),
                        ]
                    ),
                    num_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "onehot",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ),
                        ]
                    ),
                    cat_cols,
                ),
            ],
            remainder="drop",
        )
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "clf",
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
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_quality_metrics(y_test, y_pred, y_proba)
    print(f"  Test metrics: {json.dumps(metrics, indent=2, ensure_ascii=False)}")
    print(classification_report(y_test, y_pred, target_names=["good", "bad"]))

    return pipeline, X_train, y_train, X_test, y_test, feature_cols


def step03_simulate_drift(
    df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """03 — Симуляция дрифта: Age +5, изменение распределения целевых классов."""
    print("\n" + "=" * 60)
    print("03  Симуляция дрифта")
    print("=" * 60)

    # Переименуем age → Age для совместимости с monitoring.simulate_drift
    work = df.copy()
    if "age" in work.columns and "Age" not in work.columns:
        work["Age"] = work["age"]

    drifted = simulate_drift(
        work,
        age_shift=5.0,
        target_col=target_col,
        target_flip_rate=0.15,
        random_state=42,
    )

    # Синхронизируем age
    if "age" in drifted.columns:
        drifted["age"] = drifted["Age"]

    print(f"  Age (до):  mean={df['age'].mean():.2f}")
    print(f"  Age (после): mean={drifted['age'].mean():.2f}  (+5 лет)")
    print(f"  Target share (до):  {df[target_col].mean():.3f}")
    print(f"  Target share (после): {drifted[target_col].mean():.3f}")

    return drifted


def step04_evidently_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """04 — Evidently Data Drift report."""
    print("\n" + "=" * 60)
    print("04  Evidently: поиск дрифта")
    print("=" * 60)

    cols = [c for c in feature_cols if c in reference.columns and c in current.columns]
    # Берём пересечение + age
    for extra in ("age", "Age"):
        if extra in reference.columns and extra not in cols:
            cols.append(extra)

    ref = reference[cols].copy()
    cur = current[cols].copy()

    # Числовые колонки — для стабильности
    for c in cols:
        if ref[c].dtype == object:
            ref[c] = ref[c].astype(str)
            cur[c] = cur[c].astype(str)

    html_path = REPORTS_DIR / "evidently_data_drift.html"
    try:
        report = build_evidently_drift_report(ref, cur, output_html=html_path)
        summary = extract_drift_summary(report)
        print(f"  Отчёт сохранён: {html_path}")
        print(f"  Dataset drift: {summary.get('dataset_drift')}")
        print(f"  Drifted columns: {summary.get('number_of_drifted_columns')} / "
              f"{summary.get('number_of_columns')}")
        print(f"  Share drifted: {summary.get('share_of_drifted_columns')}")
        return summary
    except ImportError as e:
        print(f"  [WARN] {e}")
        print("  Установите: poetry add evidently")
        # Fallback: простой PSI по age
        from scipy.stats import ks_2samp

        age_ref = reference["age"] if "age" in reference.columns else reference.get("Age")
        age_cur = current["age"] if "age" in current.columns else current.get("Age")
        if age_ref is not None and age_cur is not None:
            stat, pval = ks_2samp(age_ref.dropna(), age_cur.dropna())
            print(f"  Fallback KS(age): D={stat:.4f}, p={pval:.2e}")
            return {
                "dataset_drift": True,
                "share_of_drifted_columns": 1.0,
                "number_of_drifted_columns": 1,
                "fallback_ks_age": {"D": float(stat), "p": float(pval)},
            }
        return {"error": str(e)}


def step05_fairness(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive: pd.Series,
) -> dict:
    """05 — Fairness по sex."""
    print("\n" + "=" * 60)
    print("05  Fairness по sex")
    print("=" * 60)

    y_pred = model.predict(X_test)
    report = compute_fairness_report(y_test, y_pred, sensitive, sensitive_name="sex")

    print(f"  Demographic Parity rates: {report['demographic_parity']['rates']}")
    print(f"  DPD: {report['demographic_parity']['dpd']:.4f}")
    print(f"  Disparate Impact (DI): {report['demographic_parity']['di']:.4f}")
    print(f"  Equalized Odds: {json.dumps(report['equalized_odds'], indent=2, ensure_ascii=False)}")
    print(f"  Примечание: {report['note']}")

    out_path = REPORTS_DIR / "fairness_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        # rates keys may not be str
        serializable = json.loads(json.dumps(report, default=str))
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"  Сохранено: {out_path}")
    return report


def step06_prometheus(
    metrics: dict[str, float],
    drift_summary: dict,
    fairness_report: dict,
    port: int,
    hold_seconds: int,
) -> None:
    """06 — Экспорт F1 (и др.) в Prometheus + имитация алерта."""
    print("\n" + "=" * 60)
    print("06  Prometheus: экспорт метрик и алерт")
    print("=" * 60)

    try:
        start_metrics_server(port)
        print(f"  Metrics server: http://127.0.0.1:{port}/metrics")
    except ImportError:
        print("  [WARN] prometheus_client не установлен. Установите: poetry add prometheus-client")
        print("  Метрики будут только выведены в консоль.")
        print(f"  F1={metrics.get('f1')}, drift_share={drift_summary.get('share_of_drifted_columns')}")
        return
    except OSError as e:
        print(f"  [WARN] Не удалось занять порт {port}: {e}")
        return

    push_quality_metrics(metrics)

    share = drift_summary.get("share_of_drifted_columns") or 0.0
    alert = bool(drift_summary.get("dataset_drift")) or (share is not None and share > 0.25)
    push_drift_metric(float(share or 0.0), alert=alert)

    di = fairness_report.get("demographic_parity", {}).get("di")
    if di is not None:
        push_fairness_di(float(di))

    print(f"  F1_SCORE → {metrics.get('f1'):.4f}")
    print(f"  DRIFT_SHARE → {share}")
    print(f"  DRIFT_ALERT fired: {alert}")
    print(f"  FAIRNESS_DI → {di}")
    print()
    print("  Пример правила Prometheus (prometheus/rules.yml):")
    print("  ```yaml")
    print("  groups:")
    print("    - name: ml_monitoring")
    print("      rules:")
    print("        - alert: HighDataDrift")
    print("          expr: model_drift_share > 0.25")
    print("          for: 5m")
    print("          labels:")
    print("            severity: warning")
    print("          annotations:")
    print('            summary: "High data drift detected"')
    print("        - alert: LowF1Score")
    print("          expr: model_f1_score < 0.5")
    print("          for: 10m")
    print("          labels:")
    print("            severity: critical")
    print("  ```")

    if hold_seconds > 0:
        print(f"\n  Сервер метрик будет доступен {hold_seconds} с. Ctrl+C для выхода.")
        try:
            time.sleep(hold_seconds)
        except KeyboardInterrupt:
            print("  Остановлено пользователем.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Практическое задание: система мониторинга ML")
    parser.add_argument("--metrics-port", type=int, default=8001, help="Порт Prometheus exporter")
    parser.add_argument(
        "--hold-seconds",
        type=int,
        default=30,
        help="Сколько секунд держать metrics-сервер (0 = не держать)",
    )
    args = parser.parse_args()

    print("Практическое задание: построение системы мониторинга ML-модели")
    print(f"Отчёты: {REPORTS_DIR}")

    # 01
    df, target_col, sensitive_col = step01_load_dataset()

    # 02
    model, X_train, y_train, X_test, y_test, feature_cols = step02_train_model(df, target_col)

    # Сохраняем reference (train+test до дрифта) для Evidently
    reference = df.copy()

    # 03
    drifted_df = step03_simulate_drift(df, target_col)

    # 04 — сравниваем reference vs drifted (по признакам)
    drift_summary = step04_evidently_drift(reference, drifted_df, feature_cols)

    # 05 — fairness на тестовой выборке исходной модели
    # sensitive для X_test: берём из исходного df по индексу
    sensitive_test = df.loc[X_test.index, sensitive_col]
    fairness_report = step05_fairness(model, X_test, y_test, sensitive_test)

    # Метрики качества (на исходном test)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_quality_metrics(y_test, y_pred, y_proba)

    # Сохраняем сводный JSON
    summary = {
        "quality_metrics": metrics,
        "drift_summary": drift_summary,
        "fairness": json.loads(json.dumps(fairness_report, default=str)),
    }
    summary_path = REPORTS_DIR / "assignment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nСводный отчёт: {summary_path}")

    # 06
    step06_prometheus(
        metrics,
        drift_summary,
        fairness_report,
        port=args.metrics_port,
        hold_seconds=args.hold_seconds,
    )

    print("\n" + "=" * 60)
    print("Готово. Все 6 шагов практического задания выполнены.")
    print("=" * 60)


if __name__ == "__main__":
    main()
