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
