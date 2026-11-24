# Churn ML Production

Рефакторинг учебного ML-проекта по прогнозированию оттока клиентов банка в production-oriented вид.

Проект включает полный цикл: обработка данных, feature engineering, обучение модели, inference API, контейнеризация, CI/CD и Kubernetes-манифесты.

## Возможности

- Модульный пайплайн обучения (`src/churn_ml`)
- FastAPI inference-сервис (`/predict`, `/health`)
- Docker + docker-compose
- GitHub Actions CI (lint, type-check, tests, сборка образа)
- Kubernetes-манифесты (Deployment + Service)
- Model Card + BPMN-диаграммы жизненного цикла модели
- Poetry, Ruff, mypy, pre-commit, pytest

## Документация

- [Model Card](docs/model_card.md) — назначение модели, данные, признаки, обучение, метрики, ограничения и рекомендации по эксплуатации.
- [BPMN: жизненный цикл модели](docs/bpmn/README.md) — workflow от загрузки данных до согласования, внедрения, мониторинга и переобучения.
- [BPMN-схема](docs/bpmn/model_lifecycle.png) — графическое представление процесса.
- [Обзор документации](docs/README.md) — единая точка входа в документацию проекта.

**Связь документов:** BPMN описывает процесс жизненного цикла, а Model Card — содержание и ограничения самой модели. Документы ссылаются друг на друга.

## Что изменено по сравнению с учебным проектом

- production-код вынесен в `src/churn_ml`;
- зависимости описаны через Poetry;
- форматирование и линтинг выполняет Ruff;
- статический анализ выполняет mypy;
- pre-commit запускает проверки автоматически перед commit;
- обучение и оценка отделены от feature engineering;
- конфигурация и random seed централизованы;
- модель сохраняется через `joblib`;
- добавлен FastAPI-сервис для инференса;
- добавлены Dockerfile и docker-compose;
- настроен CI/CD на GitHub Actions;
- добавлены базовые Kubernetes-манифесты;
- BPMN связывает подготовку данных, feature engineering, обучение, оценку, review, deployment, monitoring и цикл retraining;
- документация Model Card и BPMN связана взаимными относительными ссылками.

## Quick Start

### 1. Локально (без Docker)

```bash
# Установка зависимостей
poetry install

# Обучение модели
poetry run python -m churn_ml.train \
    --data data/raw/train.csv \
    --model-out models/churn_random_forest.joblib

# Запуск API
poetry run uvicorn churn_ml.api.main:app --reload --host 0.0.0.0 --port 8000
```

Документация API: http://localhost:8000/docs

### 2. Docker Compose (рекомендуется)

```bash
docker compose up --build
```

API будет доступен по адресу: http://localhost:8000/docs

### 3. Kubernetes (minikube / kind / k3d)

```bash
# Сборка образа
docker build -t churn-ml-api:latest .

# Деплой
kubectl apply -f k8s/

# Проброс порта
kubectl port-forward svc/churn-ml-api 8000:80
```

## API

| Метод | Эндпоинт   | Описание                              |
|-------|------------|---------------------------------------|
| GET   | `/health`  | Проверка работоспособности            |
| POST  | `/predict` | Получение прогноза оттока и вероятности |
| GET   | `/docs`    | Swagger UI                            |

### Пример запроса

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 42,
    "NumOfProducts": 1,
    "Balance": 85000.0,
    "EstimatedSalary": 100000.0,
    "IsActiveMember": 1,
    "Geography": "France",
    "Gender": "Female"
  }'
```

## Установка и проверка качества кода

```bash
poetry install
poetry run pre-commit install
```

Проверки:

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src
poetry run pytest
```

Перед первым commit можно выполнить:

```bash
poetry run pre-commit run --all-files
```

## Данные

Положите исходный `train.csv` в:

```text
data/raw/train.csv
```

Ожидается target `Exited`.

В production-версии источник данных не зашит в обучение: путь передаётся явно через CLI.

## Обучение

```bash
poetry run python -m churn_ml.train \
    --data data/raw/train.csv \
    --model-out models/churn_random_forest.joblib
```

Во время обучения выполняются:
- загрузка и очистка данных;
- feature engineering;
- train/test split;
- подбор гиперпараметров Random Forest через `HalvingRandomSearchCV`;
- расчёт метрик (`recall`, `precision`, `f1`, `roc_auc`, `average_precision`);
- сохранение модели через `joblib`.

## Важный момент про virtual environment

Само виртуальное окружение **не добавляется в Git**. В репозитории фиксируется способ его воспроизведения:

- `pyproject.toml` — зависимости и их ограничения;
- `poetry.lock` — точные версии после `poetry lock`;
- `.gitignore` — исключает `.venv/`.

После клонирования репозитория окружение создаётся заново командой:

```bash
poetry install
```

Это production-подход: бинарные файлы виртуального окружения не являются частью исходного кода.

## Структура проекта

```text
.
├── src/churn_ml/
│   ├── api/                 # FastAPI inference service
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── modeling.py
│   └── train.py
├── k8s/                     # Kubernetes manifests
├── models/                  # Trained model artifacts
├── data/
│   ├── raw/
│   └── processed/
├── docs/                    # Model Card + BPMN
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .github/workflows/ci.yml
└── README.md
```

## Tech Stack

- **ML**: scikit-learn, imbalanced-learn, pandas, numpy
- **API**: FastAPI, Uvicorn, Pydantic
- **Packaging**: Poetry
- **Quality**: Ruff, mypy, pytest, pre-commit
- **Containers**: Docker, docker-compose
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions

## Практическое задание: система мониторинга ML-модели

Реализован воспроизводимый учебный сценарий по всем шести пунктам задания:

| Шаг | Действие | Реализация |
|-----|----------|------------|
| 01 | Датасет | OpenML `credit-g` (German Credit), локальный CSV-кэш после первого запуска |
| 02 | Обучение | CatBoost при установке extra `catboost`, иначе RandomForest + sklearn Pipeline |
| 03 | Симуляция дрейфа | `age` +5 лет и инверсия 15% бинарных меток для изменения доли целевого класса |
| 04 | Evidently | `DataDriftPreset` по признакам и целевой переменной; HTML и JSON отчёты |
| 05 | Fairness | Demographic Parity, Disparate Impact и Equalized Odds по производному признаку `sex` |
| 06 | Prometheus | F1, precision, recall, ROC-AUC, drift share, label shift и fairness DI; alerting rules |

Для `credit-g` отдельного поля `sex` нет: учебная группа выводится из `personal_status`
(A91/A93/A94 → male; A92/A95 → female). Эти чувствительные поля исключены из признаков
модели и используются только для диагностического fairness-анализа. В отчёте Disparate Impact
представлен как отношение меньшей групповой доли положительных предсказаний к большей; это
симметричный скрининговый показатель, а не юридическая оценка дискриминации.

### Локальный запуск сценария

```powershell
poetry install
poetry run python scripts/practical_assignment.py --metrics-port 8001 --hold-seconds 60
```

При первом запуске требуется доступ к OpenML. Датасет сохраняется в `data/raw/credit_g_openml.csv` и используется из кэша при следующих запусках.

Артефакты формируются в `reports/practical_assignment/`:
- `evidently_data_drift.html` — визуальный Evidently-отчёт;
- `evidently_data_drift.json` — извлечённая сводка дрейфа и сдвига целевой доли;
- `fairness_report.json` — fairness по `sex`;
- `assignment_summary.json` — единая сводка эксперимента.

### Полный интеграционный тест Docker Compose

Поднимите API, monitoring exporter и Prometheus в одной Compose-сети:

```powershell
docker compose --profile monitoring up --build -d
```

Дождитесь, пока `ml-monitoring` завершит загрузку/обучение и станет healthy:

```powershell
docker compose ps
```

Затем выполните runtime smoke test (проверяет API readiness и prediction, метрики обоих exporters,
все три Prometheus scrape targets, наличие alert rules и Evidently/fairness-артефактов):

```powershell
poetry run python scripts/integration_smoke_test.py
```

Сценарий ждёт сервисы до 10 минут. Успешный результат заканчивается строкой
`INTEGRATION TEST: SUCCESS`. Он проверяет, что alert rules загружены; фактический статус
`FIRING` зависит от значений метрик и длительности `for` в правилах.

Детерминированная проверка перехода alert rules в `FIRING` запускается отдельно через `promtool`
и автоматически выполняется в GitHub Actions:

```powershell
docker run --rm `
  -v "${PWD}/monitoring:/etc/prometheus/rules:ro" `
  --entrypoint=/bin/promtool `
  prom/prometheus:v3.13.5 `
  test rules /etc/prometheus/rules/prometheus_rule_tests.yml
```

Диагностика при сбое:

```powershell
docker compose logs --tail=200 churn-ml-api ml-monitoring prometheus
```

Остановить сервисы, сохранив данные Prometheus:

```powershell
docker compose --profile monitoring down
```

В отдельном терминале можно проверить exporter:

```powershell
(Invoke-WebRequest http://localhost:8001/metrics).Content
```

### Полный контур Prometheus через Docker Compose

Для длительной работы exporter и сбора метрик Prometheus используй профиль `monitoring`:

```powershell
docker compose --profile monitoring up --build -d
```

Команда запускает:
- `churn-ml-api` — FastAPI-сервис на `http://localhost:8000`;
- `ml-monitoring` — обучение, drift/fairness-анализ и exporter на `http://localhost:8001/metrics`;
- `prometheus` — сбор метрик и вычисление alert rules на `http://localhost:9090`.

После обучения метрики появляются в Prometheus. Состояния алертов доступны на `http://localhost:9090/alerts`. Конфигурация scrape jobs находится в `monitoring/prometheus.yml`, правила — в `monitoring/prometheus_rules.yml`.

Остановить сервисы:

```powershell
docker compose --profile monitoring down
```

Сохранённые отчёты доступны локально в `reports/practical_assignment/`. Данные Prometheus сохраняются в Docker volume `prometheus-data`.

### Метрики и правила

В exporter публикуются Prometheus gauges:
- `model_f1_score`, `model_precision`, `model_recall`, `model_roc_auc`;
- `model_drift_share`;
- `model_label_positive_rate` и `model_label_positive_rate_delta`;
- `model_fairness_di`.

Настроены алерты `LowModelF1`, `ModelF1MetricMissing`, `HighDataDrift`, `TargetDistributionShift`, `MonitoringExporterDown` и `LowFairnessDisparateImpact`. Пороговые значения задаются в `monitoring/prometheus_rules.yml`; состояния видны в Prometheus UI (`/alerts`). Для отправки уведомлений во внешние каналы потребуется отдельно подключить Alertmanager и receiver.

`model_f1_score` в практическом сценарии — F1 на отложенной размеченной выборке, вычисленный при запуске задания. Это не online F1 по живому трафику: для его обновления в production необходимо собирать фактические метки/исходы по предсказаниям и пересчитывать метрики по мере поступления ground truth. Сам API экспортирует HTTP-метрики запросов через `/metrics`.

В Docker Compose каталог `./models` монтируется в контейнер API read-only. Модель должна быть доступна локально в `models/churn_random_forest.joblib`; если её нет, `/live` останется доступен, но `/ready` и `/predict` вернут ошибку о незагруженной модели.

> **Ограничения интерпретации:** `sex` выводится из `personal_status` только для учебного эксперимента. Fairness-показатели являются диагностическими сигналами, а не доказательством дискриминации или юридическим заключением. Для реальных данных необходимы законное основание обработки, минимизация данных и отдельная оценка рисков.
