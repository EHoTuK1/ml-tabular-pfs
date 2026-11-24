"""Инференс LightGBM по месячным признакам."""

from collections.abc import Sequence
from threading import Lock
from typing import cast

import numpy as np
from lightgbm import Booster
from numpy.typing import NDArray

from pfs.features.spec import CLIP, ModelType
from pfs.features.store import FeatureStore


class LightGBMPredictor:
    """Прогнозирует по совместно загруженным модели и снимку истории."""

    def __init__(self, model: Booster, features: FeatureStore) -> None:
        """Сохраняет модель и хранилище признаков для одного целевого месяца."""
        self._model = model
        self._features = features
        self._model_lock = Lock()

    def predict(self, pairs: Sequence[tuple[int, int]]) -> list[float]:
        """Возвращает обрезанные прогнозы в исходном порядке или UnknownIdError."""
        features = self._features.build(pairs, ModelType.LGBM)
        with self._model_lock:
            values = cast(
                NDArray[np.float64], self._model.predict(features, validate_features=True)
            )
        return values.clip(*CLIP).tolist()
