"""Модуль мониторинга: дрифт (Evidently), fairness, метрики Prometheus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# ---------------------------------------------------------------------------
# Симуляция дрифта
# ---------------------------------------------------------------------------


def simulate_drift(
    df: pd.DataFrame,
    age_shift: float = 5.0,
    target_col: str | None = None,
    target_flip_rate: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Симулирует covariate shift (Age) и label shift (доля целевого класса).

    Параметры соответствуют практическому заданию вебинара:
    - сдвиг age на +age_shift лет;
    - изменение распределения целевых классов (случайный flip доли меток).
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()

    if "Age" in out.columns:
        out["Age"] = out["Age"] + age_shift
    elif "age" in out.columns:
        out["age"] = out["age"] + age_shift

    if target_col and target_col in out.columns and target_flip_rate > 0:
        n = len(out)
        n_flip = int(n * target_flip_rate)
        idx = rng.choice(n, size=n_flip, replace=False)
        # Для бинарной классификации 0/1
        out.loc[out.index[idx], target_col] = 1 - out.loc[out.index[idx], target_col]

    return out


# ---------------------------------------------------------------------------
# Fairness (по полу / sex / Gender)
# ---------------------------------------------------------------------------


def demographic_parity_difference(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    sensitive: pd.Series | np.ndarray,
    privileged_value: Any = None,
) -> dict[str, float]:
    """Разница доли положительных предсказаний между группами (Demographic Parity).

    DI (Disparate Impact) = P(Ŷ=1 | unprivileged) / P(Ŷ=1 | privileged).
    DPD = P(Ŷ=1 | privileged) - P(Ŷ=1 | unprivileged).
    """
    sensitive = pd.Series(sensitive).astype(str)
    y_pred = pd.Series(y_pred).astype(int)

    groups = sensitive.unique()
    rates = {}
    for g in groups:
        mask = sensitive == g
        rates[g] = float(y_pred[mask].mean()) if mask.sum() > 0 else float("nan")

    if privileged_value is not None:
        priv = str(privileged_value)
        unpriv_groups = [g for g in groups if g != priv]
        if not unpriv_groups:
            return {"rates": rates, "dpd": float("nan"), "di": float("nan")}
        # Берём первую unprivileged для простоты отчёта
        unpriv = unpriv_groups[0]
        dpd = rates.get(priv, float("nan")) - rates.get(unpriv, float("nan"))
        di = (
            rates.get(unpriv, float("nan")) / rates[priv]
            if rates.get(priv, 0) not in (0, float("nan"))
            else float("nan")
        )
    else:
        vals = list(rates.values())
        dpd = max(vals) - min(vals) if vals else float("nan")
        di = min(vals) / max(vals) if vals and max(vals) > 0 else float("nan")

    return {"rates": rates, "dpd": float(dpd), "di": float(di)}


def equalized_odds_difference(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    sensitive: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Разница TPR и FPR между группами (упрощённый Equalized Odds)."""
    sensitive = pd.Series(sensitive).astype(str)
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

    result = {}
    for g in sensitive.unique():
        mask = sensitive == g
        yt, yp = y_true[mask], y_pred[mask]
        tp = ((yt == 1) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        tn = ((yt == 0) & (yp == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        result[g] = {"tpr": float(tpr), "fpr": float(fpr)}
    return result


def compute_fairness_report(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    sensitive: pd.Series | np.ndarray,
    sensitive_name: str = "sex",
) -> dict[str, Any]:
    """Сводный fairness-отчёт по чувствительному признаку."""
    dp = demographic_parity_difference(y_true, y_pred, sensitive)
    eo = equalized_odds_difference(y_true, y_pred, sensitive)
    return {
        "sensitive_attribute": sensitive_name,
        "demographic_parity": dp,
        "equalized_odds": eo,
        "note": (
            "Анализ fairness допустим только при наличии законного основания "
            "обработки признака (152-ФЗ). Учебный пример."
        ),
    }


# ---------------------------------------------------------------------------
# Качество модели
# ---------------------------------------------------------------------------


def compute_quality_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


# ---------------------------------------------------------------------------
# Evidently drift report
# ---------------------------------------------------------------------------


def build_evidently_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    column_mapping: dict[str, Any] | None = None,
    output_html: str | Path | None = None,
) -> Any:
    """Строит отчёт Data Drift через Evidently и опционально сохраняет HTML."""
    try:
        from evidently import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError as exc:
        raise ImportError(
            "Установите evidently: poetry add evidently  или  pip install evidently"
        ) from exc

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    if output_html is not None:
        path = Path(output_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(path))

    return report


def extract_drift_summary(report: Any) -> dict[str, Any]:
    """Извлекает краткую сводку дрифта из объекта Evidently Report."""
    try:
        as_dict = report.as_dict()
    except Exception:
        return {"error": "Не удалось сериализовать отчёт Evidently"}

    summary: dict[str, Any] = {"metrics": []}
    for metric in as_dict.get("metrics", []):
        result = metric.get("result", {})
        if "dataset_drift" in result or "number_of_drifted_columns" in result:
            summary["dataset_drift"] = result.get("dataset_drift")
            summary["share_of_drifted_columns"] = result.get("share_of_drifted_columns")
            summary["number_of_drifted_columns"] = result.get("number_of_drifted_columns")
            summary["number_of_columns"] = result.get("number_of_columns")
        summary["metrics"].append(
            {
                "metric": metric.get("metric"),
                "keys": list(result.keys())[:12],
            }
        )
    return summary


# ---------------------------------------------------------------------------
# Prometheus-метрики
# ---------------------------------------------------------------------------

_PROMETHEUS_AVAILABLE = False
try:
    from prometheus_client import Counter, Gauge, start_http_server

    F1_SCORE_GAUGE = Gauge("model_f1_score", "Current F1-score of the monitored model")
    PRECISION_GAUGE = Gauge("model_precision", "Current precision")
    RECALL_GAUGE = Gauge("model_recall", "Current recall")
    ROC_AUC_GAUGE = Gauge("model_roc_auc", "Current ROC-AUC")
    DRIFT_SHARE_GAUGE = Gauge(
        "model_drift_share",
        "Share of drifted features (Evidently)",
    )
    DRIFT_ALERTS = Counter("model_drift_alerts_total", "Total drift alert events")
    FAIRNESS_DI_GAUGE = Gauge(
        "model_fairness_di",
        "Disparate Impact ratio (fairness)",
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    F1_SCORE_GAUGE = None  # type: ignore
    PRECISION_GAUGE = None  # type: ignore
    RECALL_GAUGE = None  # type: ignore
    ROC_AUC_GAUGE = None  # type: ignore
    DRIFT_SHARE_GAUGE = None  # type: ignore
    DRIFT_ALERTS = None  # type: ignore
    FAIRNESS_DI_GAUGE = None  # type: ignore


def push_quality_metrics(metrics: dict[str, float]) -> None:
    """Обновляет Prometheus Gauge-метрики качества."""
    if not _PROMETHEUS_AVAILABLE:
        return
    if "f1" in metrics:
        F1_SCORE_GAUGE.set(metrics["f1"])
    if "precision" in metrics:
        PRECISION_GAUGE.set(metrics["precision"])
    if "recall" in metrics:
        RECALL_GAUGE.set(metrics["recall"])
    if "roc_auc" in metrics and metrics["roc_auc"] == metrics["roc_auc"]:  # not NaN
        ROC_AUC_GAUGE.set(metrics["roc_auc"])


def push_drift_metric(share_of_drifted: float, alert: bool = False) -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    DRIFT_SHARE_GAUGE.set(share_of_drifted)
    if alert:
        DRIFT_ALERTS.inc()


def push_fairness_di(di: float) -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    if di == di:  # not NaN
        FAIRNESS_DI_GAUGE.set(di)


def start_metrics_server(port: int = 8001) -> None:
    """Запускает HTTP-сервер Prometheus на указанном порту."""
    if not _PROMETHEUS_AVAILABLE:
        raise ImportError("prometheus_client не установлен")
    start_http_server(port)
