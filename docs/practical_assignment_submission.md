# Практическое задание: система мониторинга ML-модели

## Реализация по пунктам задания

| № | Требование | Где реализовано | Результат |
|---|---|---|---|
| 01 | Загрузить датасет | `scripts/practical_assignment.py`, `step01_load_dataset()` | OpenML `credit-g`; CSV-кэш `data/raw/credit_g_openml.csv` |
| 02 | Обучить модель | `step02_train_model()` | CatBoost, если установлен; иначе sklearn Pipeline с RandomForest |
| 03 | Смоделировать drift | `src/churn_ml/monitoring.py`, `simulate_drift()` | `age` сдвигается на +5 лет; 15% бинарных меток инвертируются |
| 04 | Найти drift через Evidently | `build_evidently_drift_report()`, `step04_evidently_drift()` | `DataDriftPreset`, HTML-отчёт и JSON-сводка |
| 05 | Посчитать fairness по sex | `compute_fairness_report()`, `step05_fairness()` | Demographic Parity Difference, Disparate Impact и Equalized Odds |
| 06 | Экспортировать F1 в Prometheus и настроить alert | `push_quality_metrics()`, `monitoring/prometheus_rules.yml` | `model_f1_score` и правила оповещений; доступны также drift, label shift и fairness DI |

Для `credit-g` чувствительная группа выводится из `personal_status`: A91/A93/A94 → male, A92/A95 → female. Колонки `sex` и `personal_status` исключены из признаков модели. Чувствительная группа используется только при диагностическом расчёте fairness.

## Запуск

Из корня репозитория, в Python 3.11 или 3.12:

```bash
poetry install
poetry run python scripts/practical_assignment.py --metrics-port 8001 --hold-seconds 60
```

При первом запуске нужен доступ к OpenML для загрузки `credit-g`. После выполнения сценария exporter доступен по адресу `http://localhost:8001/metrics`.

Для запуска API, monitoring exporter и Prometheus вместе:

```bash
docker compose --profile monitoring up --build -d
```

Prometheus UI: `http://localhost:9090`; список активных/ожидающих правил: `http://localhost:9090/alerts`.

## Создаваемые артефакты

В `reports/practical_assignment/`:

- `evidently_data_drift.html` — визуальный отчёт Evidently;
- `evidently_data_drift.json` — сводка data drift и label shift;
- `fairness_report.json` — fairness-метрики по `sex`;
- `assignment_summary.json` — общая сводка запуска и метрики качества модели.

## Проверки и ограничения текущей проверки

- Unit-тесты: **15 passed**.
- Интеграционный тест Evidently пропущен в текущем окружении, так как пакет `evidently` здесь не установлен. Он указан в `pyproject.toml` и `poetry.lock` (версия 0.7.23).
- Полный запуск с OpenML в текущем окружении не выполнен: сетевой доступ к `api.openml.org` отсутствует. Поэтому реальные значения F1, drift и fairness здесь не заявляются.
- Проект рассчитан на Python `>=3.11,<3.13`; текущая среда проверки использует Python 3.13. Для воспроизводимого запуска используйте Python 3.11 или 3.12 и установите зависимости через Poetry.

`model_f1_score` в этом учебном сценарии — F1 на отложенной размеченной выборке, а не непрерывно обновляемый online F1. Для production-мониторинга качества необходимо собирать фактические метки по предсказаниям и пересчитывать метрики после поступления ground truth.

Fairness-показатели служат диагностическими сигналами и сами по себе не доказывают дискриминацию или нарушение закона. Для настоящих данных обработка чувствительных атрибутов требует отдельного правового и этического обоснования.
