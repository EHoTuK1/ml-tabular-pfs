"""Контракт между API и остальной системой (Processor + Feature_store + Model).

API не знает, как считаются признаки и какая модель внутри: ему нужен объект
с методом predict. Это позволяет тестировать API с фейковым предиктором,
пока остальных компонентов нет.
"""

from collections.abc import Sequence
from typing import Protocol

from pfs.errors import UnknownIdError

__all__ = ["Predictor", "UnknownIdError"]


class Predictor(Protocol):
    def predict(self, pairs: Sequence[tuple[int, int]]) -> list[float]:
        """Прогноз item_cnt_month для пар (shop_id, item_id), порядок сохраняется.

        Прогноз уже обрезан (clip) так же, как в офлайн-оценке.
        Бросает UnknownIdError, если среди id есть неизвестные.
        """
        ...
