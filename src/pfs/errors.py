"""Исключения, общие для компонентов сервиса."""

from collections.abc import Iterable


class UnknownIdError(ValueError):
    """Магазин или товар отсутствует в справочниках (холодный старт по id)."""

    def __init__(self, shop_ids: Iterable[int] = (), item_ids: Iterable[int] = ()):
        self.shop_ids = sorted({int(s) for s in shop_ids})
        self.item_ids = sorted({int(i) for i in item_ids})
        super().__init__(f"unknown shop_id: {self.shop_ids}, unknown item_id: {self.item_ids}")
