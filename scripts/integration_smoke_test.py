#!/usr/bin/env python3
"""End-to-end smoke test for the running Churn ML Monitoring Compose stack.

Start the stack first:
    docker compose --profile monitoring up --build -d

Then run:
    poetry run python scripts/integration_smoke_test.py

The test checks API readiness and inference, both Prometheus exporters, scrape
health, loaded alert rules, and the Evidently/fairness report artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = ROOT / "reports" / "practical_assignment"


class IntegrationTestError(RuntimeError):
    """Raised when an integration assertion fails."""


def _request_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise IntegrationTestError(f"GET {url} failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise IntegrationTestError(f"GET {url} returned a non-object JSON response")
    return payload


def _request_text(url: str, timeout: float = 5.0) -> str:
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise IntegrationTestError(f"GET {url} failed: {exc}") from exc


def _post_json(url: str, body: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise IntegrationTestError(f"POST {url} failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise IntegrationTestError(f"POST {url} returned a non-object JSON response")
    return payload


def _wait_for(
    description: str,
    operation: Callable[[], Any],
    predicate: Callable[[Any], bool],
    timeout_seconds: int,
    interval_seconds: float,
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            result = operation()
            if predicate(result):
                print(f"[PASS] {description}")
                return result
        except Exception as exc:  # Retry while containers initialise.
            last_error = exc
        time.sleep(interval_seconds)
    detail = f" Last error: {last_error}" if last_error else ""
    raise IntegrationTestError(f"Timed out waiting for {description}.{detail}")


def _assert_metric_present(exposition: str, metric_name: str) -> None:
    if not any(
        line.startswith(f"{metric_name} ") or line.startswith(f"{metric_name}{{")
        for line in exposition.splitlines()
    ):
        raise IntegrationTestError(f"Prometheus metric {metric_name!r} is missing")


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise IntegrationTestError(f"Expected report artifact is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationTestError(f"Cannot read JSON report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise IntegrationTestError(f"Report {path} must contain a JSON object")
    return payload


def _prometheus_query(prometheus_url: str, expression: str) -> dict[str, Any]:
    query = urlencode({"query": expression})
    payload = _request_json(f"{prometheus_url}/api/v1/query?{query}")
    if payload.get("status") != "success":
        raise IntegrationTestError(f"Prometheus query failed: {expression}")
    result = payload.get("data", {}).get("result", [])
    if not result:
        raise IntegrationTestError(f"Prometheus query returned no samples: {expression}")
    return payload


def run_checks(
    api_url: str,
    exporter_url: str,
    prometheus_url: str,
    reports_dir: Path,
    timeout_seconds: int,
    interval_seconds: float,
) -> None:
    print("\nChurn ML Monitoring — integration smoke test")
    print(f"API:         {api_url}")
    print(f"Exporter:    {exporter_url}")
    print(f"Prometheus:  {prometheus_url}")
    print(f"Reports:     {reports_dir}\n")

    _wait_for(
        "FastAPI liveness (/live)",
        lambda: _request_json(f"{api_url}/live"),
        lambda value: value.get("status") == "alive",
        timeout_seconds,
        interval_seconds,
    )
    ready = _wait_for(
        "FastAPI model readiness (/ready)",
        lambda: _request_json(f"{api_url}/ready"),
        lambda value: value.get("status") == "ready" and value.get("model_loaded") == "true",
        timeout_seconds,
        interval_seconds,
    )
    if ready.get("model_loaded") != "true":
        raise IntegrationTestError("FastAPI reports that the model is not loaded")

    prediction = _post_json(
        f"{api_url}/predict",
        {
            "Age": 42,
            "NumOfProducts": 1,
            "Balance": 85000.0,
            "EstimatedSalary": 100000.0,
            "IsActiveMember": 1,
            "Geography": "France",
            "Gender": "Female",
        },
    )
    if prediction.get("prediction") not in (0, 1):
        raise IntegrationTestError(f"Unexpected prediction payload: {prediction}")
    probability = prediction.get("probability")
    if not isinstance(probability, int | float) or not 0.0 <= float(probability) <= 1.0:
        raise IntegrationTestError(f"Unexpected prediction probability: {probability}")
    print("[PASS] FastAPI inference (/predict)")

    api_metrics = _request_text(f"{api_url}/metrics")
    _assert_metric_present(api_metrics, "churn_predict_total")
    _assert_metric_present(api_metrics, "churn_predict_latency_seconds_count")
    print("[PASS] FastAPI inference metrics (/metrics)")

    exporter_metrics = _wait_for(
        "ML monitoring exporter (/metrics)",
        lambda: _request_text(f"{exporter_url}/metrics"),
        lambda value: "model_f1_score" in value,
        timeout_seconds,
        interval_seconds,
    )
    for metric in (
        "model_f1_score",
        "model_precision",
        "model_recall",
        "model_roc_auc",
        "model_drift_share",
        "model_fairness_di",
        "model_label_positive_rate",
        "model_label_positive_rate_delta",
    ):
        _assert_metric_present(exporter_metrics, metric)
    print("[PASS] Quality, drift, label-shift and fairness metrics exported")

    prometheus_health = _wait_for(
        "Prometheus readiness (/-/ready)",
        lambda: _request_text(f"{prometheus_url}/-/ready"),
        lambda value: "ready" in value.lower(),
        timeout_seconds,
        interval_seconds,
    )
    if not prometheus_health:
        raise IntegrationTestError("Prometheus readiness endpoint returned an empty response")

    expected_jobs = {"prometheus", "churn-ml-api", "ml-monitoring-assignment"}

    def _targets_up() -> dict[str, str]:
        payload = _request_json(f"{prometheus_url}/api/v1/targets?state=active")
        if payload.get("status") != "success":
            raise IntegrationTestError("Prometheus targets API did not return success")
        targets = payload.get("data", {}).get("activeTargets", [])
        return {
            target.get("labels", {}).get("job", ""): target.get("health", "unknown")
            for target in targets
        }

    target_states = _wait_for(
        "Prometheus scrape targets all UP",
        _targets_up,
        lambda states: expected_jobs.issubset(states)
        and all(states[job] == "up" for job in expected_jobs),
        timeout_seconds,
        interval_seconds,
    )
    print(f"       Scrape target states: {target_states}")

    for metric in (
        "model_f1_score",
        "model_drift_share",
        "model_fairness_di",
        "churn_predict_total",
    ):
        _wait_for(
            f"Prometheus query returns {metric}",
            lambda metric_name=metric: _prometheus_query(prometheus_url, metric_name),
            lambda payload: bool(payload.get("data", {}).get("result")),
            timeout_seconds,
            interval_seconds,
        )
    print("[PASS] Prometheus stores API and ML monitoring samples")

    rules_payload = _request_json(f"{prometheus_url}/api/v1/rules")
    if rules_payload.get("status") != "success":
        raise IntegrationTestError("Prometheus rules API did not return success")
    loaded_rules = {
        rule.get("name")
        for group in rules_payload.get("data", {}).get("groups", [])
        for rule in group.get("rules", [])
    }
    expected_rules = {
        "LowModelF1",
        "ModelF1MetricMissing",
        "HighDataDrift",
        "TargetDistributionShift",
        "MonitoringExporterDown",
        "LowFairnessDisparateImpact",
    }
    missing_rules = expected_rules - loaded_rules
    if missing_rules:
        raise IntegrationTestError(f"Prometheus alert rules not loaded: {sorted(missing_rules)}")
    print(f"[PASS] Prometheus loaded all {len(expected_rules)} alert rules")

    summary = _read_json_file(reports_dir / "assignment_summary.json")
    drift_report = _read_json_file(reports_dir / "evidently_data_drift.json")
    fairness_report = _read_json_file(reports_dir / "fairness_report.json")
    html_report = reports_dir / "evidently_data_drift.html"
    if not html_report.is_file() or html_report.stat().st_size == 0:
        raise IntegrationTestError(f"Evidently HTML report is missing or empty: {html_report}")

    quality = summary.get("quality_metrics", {})
    if not isinstance(quality.get("f1"), int | float):
        raise IntegrationTestError("assignment_summary.json has no numeric F1 metric")
    drift_validation = summary.get("drift_validation", {})
    age_delta = drift_validation.get("age_mean_delta")
    if not isinstance(age_delta, int | float) or abs(float(age_delta) - 5.0) > 1e-8:
        raise IntegrationTestError(f"Expected +5.0 years age shift; got {age_delta!r}")
    label_distribution = drift_report.get("label_distribution", {})
    if not isinstance(label_distribution.get("positive_rate_delta"), int | float):
        raise IntegrationTestError("Evidently summary has no numeric target-distribution shift")
    demographic_parity = fairness_report.get("demographic_parity", {})
    if not isinstance(demographic_parity.get("rates"), dict):
        raise IntegrationTestError("Fairness report has no per-group positive prediction rates")
    print("[PASS] Evidently HTML/JSON and fairness JSON artifacts")
    print(f"       F1={quality['f1']:.4f}; age mean delta={age_delta:+.1f}")
    print(
        "       target positive-rate delta="
        f"{label_distribution['positive_rate_delta']:+.4f}; "
        f"sex groups={list(demographic_parity['rates'])}"
    )

    print("\nINTEGRATION TEST: SUCCESS")
    print(
        "All runtime checks passed. Alert rules are loaded; firing depends on live metric values and for: durations."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--exporter-url", default="http://127.0.0.1:8001")
    parser.add_argument("--prometheus-url", default="http://127.0.0.1:9090")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    args = parser.parse_args()

    if args.timeout_seconds <= 0 or args.interval_seconds <= 0:
        parser.error("Timeout and polling interval must be positive")

    try:
        run_checks(
            api_url=args.api_url.rstrip("/"),
            exporter_url=args.exporter_url.rstrip("/"),
            prometheus_url=args.prometheus_url.rstrip("/"),
            reports_dir=args.reports_dir,
            timeout_seconds=args.timeout_seconds,
            interval_seconds=args.interval_seconds,
        )
    except IntegrationTestError as exc:
        print(f"\nINTEGRATION TEST: FAILED\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
