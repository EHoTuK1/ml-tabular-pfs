"""Контракт между API и остальной системой (Processor + Feature_store + Model).

API не знает, как считаются признаки и какая модель внутри: ему нужен объект
с методом predict. Это позволяет тестировать API с фейковым предиктором,
пока остальных компонентов нет.
"""

from collections.abc import Sequence
from typing import Protocol


class UnknownIdError(ValueError):
    """Магазин или товар отсутствует в справочниках (холодный старт по id)."""

    def __init__(self, shop_ids: Sequence[int] = (), item_ids: Sequence[int] = ()):
        self.shop_ids = sorted(set(shop_ids))
        self.item_ids = sorted(set(item_ids))
        super().__init__(f"unknown shop_id: {self.shop_ids}, unknown item_id: {self.item_ids}")


class Predictor(Protocol):
    def predict(self, pairs: Sequence[tuple[int, int]]) -> list[float]:
        """Прогноз item_cnt_month для пар (shop_id, item_id), порядок сохраняется.

        Прогноз уже обрезан (clip) так же, как в офлайн-оценке.
        Бросает UnknownIdError, если среди id есть неизвестные.
        """
        ...
