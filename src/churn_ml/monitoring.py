"""Модуль мониторинга: дрифт (Evidently), fairness, метрики Prometheus."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

if TYPE_CHECKING:
    from prometheus_client import Counter, Gauge

# ---------------------------------------------------------------------------
# Симуляция дрифта
# ---------------------------------------------------------------------------


class DemographicParityReport(TypedDict):
    """Результат расчёта Demographic Parity."""

    rates: dict[str, float]
    dpd: float
    di: float


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

    for age_column in ("age", "Age"):
        if age_column in out.columns:
            out[age_column] = out[age_column] + age_shift

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
) -> DemographicParityReport:
    """Разница доли положительных предсказаний между группами (Demographic Parity).

    При заданном privileged_value DI = rate(unprivileged) / rate(privileged).
    Без baseline используется симметричное отношение min(rate) / max(rate).
    DPD без baseline — размах положительных prediction rates между группами.
    """
    sensitive = pd.Series(sensitive).reset_index(drop=True).astype(str)
    y_pred = pd.Series(y_pred).reset_index(drop=True).astype(int)

    groups = sensitive.unique()
    rates: dict[str, float] = {}
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
    sensitive = pd.Series(sensitive).reset_index(drop=True).astype(str)
    y_true = pd.Series(y_true).reset_index(drop=True).astype(int)
    y_pred = pd.Series(y_pred).reset_index(drop=True).astype(int)

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
    """Строит Evidently Data Drift report и сохраняет HTML-отчёт.

    Совместимо с API Evidently 0.7.x (Snapshot возвращается из ``Report.run``).
    """
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        raise ImportError("Установите зависимости проекта: poetry install") from exc

    report = Report(metrics=[DataDriftPreset()])
    run_kwargs: dict[str, Any] = {
        "current_data": current,
        "reference_data": reference,
    }
    if column_mapping is not None:
        run_kwargs["column_mapping"] = column_mapping

    try:
        snapshot = report.run(**run_kwargs)
    except TypeError:
        # Evidently 0.7 removed/changed some legacy run arguments.
        run_kwargs.pop("column_mapping", None)
        snapshot = report.run(**run_kwargs)

    if output_html is not None:
        path = Path(output_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(path))

    return snapshot


def _snapshot_to_dict(snapshot: Any) -> dict[str, Any]:
    """Преобразует Evidently Snapshot или обычный mapping в dict."""
    if isinstance(snapshot, dict):
        return snapshot

    for method_name in ("dict", "as_dict"):
        method = getattr(snapshot, method_name, None)
        if callable(method):
            try:
                payload = method()
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload

    json_method = getattr(snapshot, "json", None)
    if callable(json_method):
        import json

        with suppress(TypeError, ValueError):
            payload = json.loads(json_method())
            if isinstance(payload, dict):
                return payload

    return {}


def extract_drift_summary(report: Any) -> dict[str, Any]:
    """Извлекает сводку из Evidently Snapshot разных поддерживаемых версий."""
    payload = _snapshot_to_dict(report)
    if not payload:
        return {"error": "Не удалось сериализовать отчёт Evidently"}

    summary: dict[str, Any] = {"metrics": []}
    drifted_columns: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "dataset_drift" in value:
                summary["dataset_drift"] = value["dataset_drift"]
            if "share_of_drifted_columns" in value:
                summary["share_of_drifted_columns"] = value["share_of_drifted_columns"]
            if "number_of_drifted_columns" in value:
                summary["number_of_drifted_columns"] = value["number_of_drifted_columns"]
            if "number_of_columns" in value:
                summary["number_of_columns"] = value["number_of_columns"]

            drift_flag = value.get("drift_detected")
            column_name = value.get("column_name") or value.get("column")
            if drift_flag is True and column_name is not None:
                drifted_columns.append(str(column_name))

            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
    summary["metrics"] = payload.get("metrics", [])
    metrics = payload.get("metrics", [])
    if isinstance(metrics, list):
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            metric_id = str(metric.get("metric_id") or metric.get("metric") or "")
            result = metric.get("result", {})
            if "DriftedColumnsCount" in metric_id and isinstance(result, dict):
                summary.setdefault("number_of_drifted_columns", result.get("count"))
                summary.setdefault("share_of_drifted_columns", result.get("share"))
                summary.setdefault("number_of_columns", result.get("total"))

    if drifted_columns:
        summary["drifted_columns"] = sorted(set(drifted_columns))

    count = summary.get("number_of_drifted_columns")
    total = summary.get("number_of_columns")
    share = summary.get("share_of_drifted_columns")
    if share is None and count is not None and total:
        share = float(count) / float(total)
        summary["share_of_drifted_columns"] = share
    if share is None and drifted_columns:
        total_columns = summary.get("number_of_columns")
        if total_columns:
            share = len(set(drifted_columns)) / float(total_columns)
            summary["share_of_drifted_columns"] = share
    if "dataset_drift" not in summary:
        summary["dataset_drift"] = bool(share is not None and float(share) >= 0.5)

    return summary


def compute_label_distribution_shift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_col: str,
    positive_label: Any = 1,
) -> dict[str, float]:
    """Сравнивает долю положительного класса в reference и current выборках."""
    if target_col not in reference.columns or target_col not in current.columns:
        raise ValueError(f"Целевая колонка {target_col!r} отсутствует в одной из выборок")
    if reference.empty or current.empty:
        raise ValueError("Для расчёта label shift обе выборки должны быть непустыми")

    reference_rate = float((reference[target_col] == positive_label).mean())
    current_rate = float((current[target_col] == positive_label).mean())
    return {
        "reference_positive_rate": reference_rate,
        "current_positive_rate": current_rate,
        "positive_rate_delta": current_rate - reference_rate,
    }


# ---------------------------------------------------------------------------
# Prometheus-метрики
# ---------------------------------------------------------------------------

_PROMETHEUS_AVAILABLE = False

F1_SCORE_GAUGE: Gauge | None
PRECISION_GAUGE: Gauge | None
RECALL_GAUGE: Gauge | None
ROC_AUC_GAUGE: Gauge | None
DRIFT_SHARE_GAUGE: Gauge | None
DRIFT_ALERTS: Counter | None
FAIRNESS_DI_GAUGE: Gauge | None
LABEL_POSITIVE_RATE_GAUGE: Gauge | None
LABEL_SHIFT_DELTA_GAUGE: Gauge | None

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
    LABEL_POSITIVE_RATE_GAUGE = Gauge(
        "model_label_positive_rate",
        "Current positive-class share in the monitored data",
    )
    LABEL_SHIFT_DELTA_GAUGE = Gauge(
        "model_label_positive_rate_delta",
        "Current minus reference positive-class share",
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    F1_SCORE_GAUGE = None
    PRECISION_GAUGE = None
    RECALL_GAUGE = None
    ROC_AUC_GAUGE = None
    DRIFT_SHARE_GAUGE = None
    DRIFT_ALERTS = None
    FAIRNESS_DI_GAUGE = None
    LABEL_POSITIVE_RATE_GAUGE = None
    LABEL_SHIFT_DELTA_GAUGE = None


def push_quality_metrics(metrics: dict[str, float]) -> None:
    """Обновляет Prometheus Gauge-метрики качества."""
    if not _PROMETHEUS_AVAILABLE:
        return
    assert F1_SCORE_GAUGE is not None
    assert PRECISION_GAUGE is not None
    assert RECALL_GAUGE is not None
    assert ROC_AUC_GAUGE is not None
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

    assert DRIFT_SHARE_GAUGE is not None
    assert DRIFT_ALERTS is not None
    DRIFT_SHARE_GAUGE.set(share_of_drifted)
    if alert:
        DRIFT_ALERTS.inc()


def push_fairness_di(di: float) -> None:
    if not _PROMETHEUS_AVAILABLE:
        return

    assert FAIRNESS_DI_GAUGE is not None
    if di == di:  # not NaN
        FAIRNESS_DI_GAUGE.set(di)


def start_metrics_server(port: int = 8001) -> None:
    """Запускает HTTP-сервер Prometheus на указанном порту."""
    if not _PROMETHEUS_AVAILABLE:
        raise ImportError("prometheus_client не установлен")
    start_http_server(port)


def push_label_distribution_metrics(positive_rate: float, delta: float) -> None:
    """Экспортирует текущую долю положительного класса и её сдвиг."""
    if not _PROMETHEUS_AVAILABLE:
        return

    assert LABEL_POSITIVE_RATE_GAUGE is not None
    assert LABEL_SHIFT_DELTA_GAUGE is not None
    LABEL_POSITIVE_RATE_GAUGE.set(positive_rate)
    LABEL_SHIFT_DELTA_GAUGE.set(delta)
