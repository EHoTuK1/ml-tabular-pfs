"""FastAPI-приложение: /health и /predict."""

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status

from pfs.api.predictor import Predictor, UnknownIdError
from pfs.api.schemas import HealthResponse, Prediction, PredictRequest, PredictResponse


def create_app(predictor: Predictor | None = None) -> FastAPI:
    """Фабрика приложения. predictor подставляется снаружи (в тестах — фейк)."""
    app = FastAPI(title="PFS sales forecast", version="0.1.0")
    app.state.predictor = predictor

    def get_predictor(request: Request) -> Predictor:
        p = request.app.state.predictor
        if p is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model is not loaded")
        return p

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        # Сервис отвечает всегда; готовность к прогнозам видна по model_loaded.
        return HealthResponse(status="ok", model_loaded=request.app.state.predictor is not None)

    # Обычный def, не async: расчёт признаков и LightGBM грузят CPU,
    # FastAPI выполнит такую функцию в пуле потоков и не заблокирует сервер.
    @app.post("/predict", response_model=PredictResponse)
    def predict(
        body: PredictRequest, predictor: Annotated[Predictor, Depends(get_predictor)]
    ) -> PredictResponse:
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

    return app
