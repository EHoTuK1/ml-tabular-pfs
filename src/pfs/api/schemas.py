"""Схемы запросов и ответов. Pydantic проверяет типы, ошибки ввода -> 422."""

from pydantic import BaseModel, Field

from pfs.training import DEFAULT_LEARNING_RATE, DEFAULT_N_ESTIMATORS, DEFAULT_NUM_LEAVES

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


class TrainRequest(BaseModel):
    """Гиперпараметры запуска LightGBM с офлайн-значениями по умолчанию."""

    learning_rate: float = Field(default=DEFAULT_LEARNING_RATE, gt=0, allow_inf_nan=False)
    num_leaves: int = Field(default=DEFAULT_NUM_LEAVES, ge=2, le=131072)
    n_estimators: int = Field(default=DEFAULT_N_ESTIMATORS, ge=1)


class TrainResponse(BaseModel):
    """Идентификатор принятого запуска обучения."""

    run_id: str
