from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from churn_ml.features import add_features

app = FastAPI(
    title="Churn Prediction API",
    description="API для прогнозирования оттока клиентов банка (+ Prometheus /metrics)",
    version="1.1.0",
)

MODEL_PATH = Path("models/churn_random_forest.joblib")
model = None

# Prometheus (опционально)
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    PREDICT_COUNTER = Counter(
        "churn_predict_total",
        "Total prediction requests",
        ["outcome"],
    )
    PREDICT_LATENCY = Histogram(
        "churn_predict_latency_seconds",
        "Prediction latency in seconds",
    )
    _PROM_OK = True
except ImportError:
    _PROM_OK = False


class PredictRequest(BaseModel):
    """Исходные признаки клиента (до feature engineering)."""

    Age: float
    NumOfProducts: int
    Balance: float
    EstimatedSalary: float
    IsActiveMember: int = Field(..., description="0 или 1")
    Geography: str = Field(..., description="France / Germany / Spain")
    Gender: str = Field(..., description="Male / Female")


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="0 — останется, 1 — уйдёт")
    probability: float = Field(..., description="Вероятность оттока")


@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        # Не падаем при старте без модели — удобно для /metrics и /health
        print(f"Предупреждение: модель не найдена ({MODEL_PATH}). /predict будет недоступен.")
        return
    model = joblib.load(MODEL_PATH)
    print(f"Модель успешно загружена из {MODEL_PATH}")


@app.get("/health")
def health() -> dict[str, str]:
    status = "ok" if model is not None else "degraded"
    return {"status": status, "model_loaded": str(model is not None)}


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus exposition endpoint."""
    if not _PROM_OK:
        return Response(
            content="# prometheus_client not installed\n",
            media_type="text/plain",
        )
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    data = pd.DataFrame([request.model_dump()])
    data = add_features(data)

    try:
        if _PROM_OK:
            with PREDICT_LATENCY.time():
                proba = model.predict_proba(data)[0][1]
                pred = int(proba >= 0.5)
            PREDICT_COUNTER.labels(outcome=str(pred)).inc()
        else:
            proba = model.predict_proba(data)[0][1]
            pred = int(proba >= 0.5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}") from e

    return PredictResponse(prediction=pred, probability=round(float(proba), 4))
