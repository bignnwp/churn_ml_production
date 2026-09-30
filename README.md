# Churn ML Production

Рефакторинг учебного ML-проекта по прогнозированию оттока клиентов банка.

## Документация

- [Model Card](docs/model_card.md) — назначение модели, данные, признаки, обучение, метрики, ограничения и рекомендации по эксплуатации.
- [BPMN: жизненный цикл модели](docs/bpmn/README.md) — workflow от загрузки данных до согласования, внедрения, мониторинга и переобучения.
- [BPMN-схема](docs/bpmn/model_lifecycle.png) — графическое представление процесса.
- [Обзор документации](docs/README.md) — единая точка входа в документацию проекта.

**Связь документов:** BPMN описывает процесс жизненного цикла, а Model Card — содержание и ограничения самой модели. Документы ссылаются друг на друга.

## Что изменено

- production-код вынесен в `src/churn_ml`;
- зависимости описаны через Poetry;
- форматирование и линтинг выполняет Ruff;
- статический анализ выполняет mypy;
- pre-commit запускает проверки автоматически перед commit;
- обучение и оценка отделены от feature engineering;
- конфигурация и random seed централизованы;
- модель сохраняется через `joblib`;
- BPMN связывает подготовку данных, feature engineering, обучение, оценку, review, deployment, monitoring и цикл retraining;
- документация Model Card и BPMN связана взаимными относительными ссылками.

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

## Установка

```bash
poetry install
poetry run pre-commit install
```

Проверка:

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

Во время обучения выполняются загрузка и очистка данных, feature engineering, train/test split, подбор гиперпараметров Random Forest через `HalvingRandomSearchCV` и расчёт `recall`, `precision`, `f1`, `roc_auc` и `average_precision`.

## Структура

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── bpmn/
│   │   ├── model_lifecycle.png
│   │   └── README.md
│   ├── model_card.md
│   └── README.md
├── models/
├── reports/
├── scripts/
├── src/
│   └── churn_ml/
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── modeling.py
│       └── train.py
├── tests/
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md
```
