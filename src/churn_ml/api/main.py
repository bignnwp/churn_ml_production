from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn_ml.features import add_features

app = FastAPI(
    title="Churn Prediction API",
    description="API для прогнозирования оттока клиентов банка",
    version="1.0.0",
)

MODEL_PATH = Path("models/churn_random_forest.joblib")
model = None


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
        raise RuntimeError(f"Модель не найдена: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"Модель успешно загружена из {MODEL_PATH}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    # Превращаем запрос в DataFrame
    data = pd.DataFrame([request.model_dump()])

    # Применяем тот же feature engineering, что использовался при обучении
    data = add_features(data)

    try:
        proba = model.predict_proba(data)[0][1]
        pred = int(proba >= 0.5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")

    return PredictResponse(
        prediction=pred,
        probability=round(float(proba), 4)
    )