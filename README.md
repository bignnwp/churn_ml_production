# Churn ML Production

Рефакторинг учебного ML-проекта по прогнозированию оттока клиентов банка.

## Что изменено

- notebook оставлен как исследовательский артефакт;
- production-код вынесен в `src/churn_ml`;
- зависимости описаны через Poetry;
- форматирование и линтинг выполняет Ruff;
- статический анализ выполняет mypy;
- pre-commit запускает проверки автоматически перед commit;
- обучение и оценка отделены от feature engineering;
- конфигурация и random seed централизованы;
- модель сохраняется через `joblib`.

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

Исходный notebook загружал датасет из GitHub. В production-версии источник данных не зашит в обучение: путь передаётся явно через CLI.

## Обучение

```bash
poetry run python -m churn_ml.train \
    --data data/raw/train.csv \
    --model-out models/churn_random_forest.joblib
```

## Структура

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
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
