"""HTTP API для прогнозов и запусков обучения."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status

from pfs.api.predictor import Predictor, UnknownIdError
from pfs.api.schemas import (
    HealthResponse,
    Prediction,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from pfs.processor import Processor, TrainingInProgressError, TrainingStatus


def create_app(
    predictor: Predictor | None = None, *, processor: Processor | None = None
) -> FastAPI:
    """Создаёт приложение с подставляемым предиктором и опциональным обучением."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        """Загружает активную версию и завершает фоновое обучение при остановке."""
        if processor is not None:
            processor.restore()
        try:
            yield
        finally:
            if processor is not None:
                processor.close()

    app = FastAPI(title="PFS sales forecast", version="0.1.0", lifespan=lifespan)
    app.state.predictor = predictor

    def active_predictor(request: Request) -> Predictor | None:
        """Берёт текущую версию модели или предиктор, заданный при создании API."""
        if processor is not None and processor.model_loaded:
            return processor
        return request.app.state.predictor

    def get_predictor(request: Request) -> Predictor:
        """Возвращает HTTP 503, если готового предиктора ещё нет."""
        p = active_predictor(request)
        if p is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model is not loaded")
        return p

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Сообщает о доступности сервиса и готовности модели."""
        return HealthResponse(status="ok", model_loaded=active_predictor(request) is not None)

    # Обычный def, не async: расчёт признаков и LightGBM грузят CPU,
    # FastAPI выполнит такую функцию в пуле потоков и не заблокирует сервер.
    @app.post("/predict", response_model=PredictResponse)
    def predict(
        body: PredictRequest, predictor: Annotated[Predictor, Depends(get_predictor)]
    ) -> PredictResponse:
        """Возвращает прогнозы в порядке запроса или ошибку неизвестных ID."""
        pairs = [(it.shop_id, it.item_id) for it in body.items]
        try:
            values = predictor.predict(pairs)
        except UnknownIdError as e:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                {"message": "unknown ids", "shop_id": e.shop_ids, "item_id": e.item_ids},
            ) from e
        if len(values) != len(pairs):
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, "predictor returned wrong length"
            )
        return PredictResponse(
            predictions=[
                Prediction(shop_id=s, item_id=i, item_cnt_month=v)
                for (s, i), v in zip(pairs, values, strict=True)
            ]
        )

    if processor is not None:

        @app.post("/train", status_code=status.HTTP_202_ACCEPTED)
        def start_training(body: TrainRequest) -> TrainResponse:
            """Запускает обучение или возвращает HTTP 409 при конфликте."""
            try:
                run_id = processor.start_training(
                    learning_rate=body.learning_rate,
                    num_leaves=body.num_leaves,
                    n_estimators=body.n_estimators,
                )
            except TrainingInProgressError as error:
                raise HTTPException(
                    status.HTTP_409_CONFLICT, "training already in progress"
                ) from error
            return TrainResponse(run_id=run_id)

        @app.get("/train/{run_id}", response_model_exclude_none=True)
        def training_status(run_id: str) -> TrainingStatus:
            """Возвращает состояние обучения или HTTP 404 для неизвестного запуска."""
            run = processor.get_training_status(run_id)
            if run is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "training run not found")
            return run

    return app
