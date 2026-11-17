"""Схемы запросов и ответов. Pydantic проверяет типы, ошибки ввода -> 422."""

from pydantic import BaseModel, Field

# Верхняя граница на размер пакета: защищает память сервиса от огромных запросов.
# Test Kaggle содержит 214 200 пар, берём с запасом.
MAX_ITEMS = 250_000


class Pair(BaseModel):
    shop_id: int = Field(ge=0)
    item_id: int = Field(ge=0)


class PredictRequest(BaseModel):
    items: list[Pair] = Field(min_length=1, max_length=MAX_ITEMS)


class Prediction(Pair):
    item_cnt_month: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
