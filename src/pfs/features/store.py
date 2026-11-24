"""Feature_store: история до месяца прогноза и расчёт признаков по запросу.

При создании история (сетка и транзакции) сжимается в таблицы поиска: таргет
пар за нужные лаги, средние по уровням за m-1, первые/последние продажи, цены.
По запросу признаки собираются векторно поиском по этим таблицам. Формулы те же,
что в tmp/scripts/feature_engineering.py; совпадение проверяет тест паритета.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pfs.errors import UnknownIdError
from pfs.features.gold import SHOP_MAP
from pfs.features.spec import FEATURES, ModelType

TARGET_LAGS = (1, 2, 3, 4, 5, 6, 12)
ROLL_WINDOWS = (3, 6)
# Ключ пары из двух id в одном int64 (id < KEY_MUL), так поиск идёт по одному индексу.
KEY_MUL = 1_000_000

Ints = NDArray[np.integer[Any]]
Floats = NDArray[np.float64]


def _key(a: Ints, b: Ints) -> Ints:
    return a * KEY_MUL + b


def _lookup(table: "pd.Series[float]", keys: Ints) -> Floats:
    """Значения table по ключам; отсутствующий ключ -> NaN."""
    pos = table.index.get_indexer(pd.Index(keys))
    values = table.to_numpy(dtype=np.float64)
    if len(values) == 0:
        return np.full(len(keys), np.nan)
    return np.where(pos >= 0, values[pos], np.nan)


def _series(keys: Ints, values: NDArray[np.float64]) -> "pd.Series[float]":
    return pd.Series(values, index=pd.Index(keys), dtype="float64")


def _category_codes(categories: pd.DataFrame) -> pd.DataFrame:
    """category_type_id и item_type_id по item_category_id (как add_categories)."""
    parts = categories["item_category_name"].str.split("-", n=1, expand=True)
    return pd.DataFrame(
        {
            "category_type_id": parts[0].str.strip().rank(method="dense").astype("uint8"),
            "item_type_id": parts[1].str.strip().rank(method="dense").fillna(0).astype("uint8"),
        }
    ).set_index(categories["item_category_id"].astype("int64"))


def _city_codes(shops: pd.DataFrame) -> "pd.Series[int]":
    """city_id по shop_id (как add_cities)."""
    other = shops["shop_name"].str.contains(
        "Интернет-магазин|Цифровой склад|Выездная Торговля", na=False, case=False
    )
    city = (
        shops["shop_name"]
        .str.replace("!", "", regex=False)
        .str.strip()
        .mask(other, "Прочее")
        .str.split(n=1)
        .str[0]
    )
    codes = city.rank(method="dense").fillna(0).astype("uint8")
    return pd.Series(codes.to_numpy(), index=shops["shop_id"].astype("int64"))


class FeatureStore:
    """Признаки для пар (shop_id, item_id) в месяце target_month.

    Использует только историю строго до target_month, как офлайн-расчёт.
    """

    def __init__(
        self,
        grid: pd.DataFrame,
        sales: pd.DataFrame,
        items: pd.DataFrame,
        categories: pd.DataFrame,
        shops: pd.DataFrame,
        *,
        target_month: int,
    ) -> None:
        """grid: месячная сетка (date_block_num, shop_id, item_id, target), магазины объединены.
        sales: транзакции (date_block_num, item_id, item_price, item_cnt_day).
        items, categories, shops: справочники Kaggle.
        Строки grid и sales с месяцем >= target_month отбрасываются.
        """
        m = target_month
        self.target_month = m

        # справочники
        self._shops = pd.Index(shops["shop_id"].astype("int64").unique())
        self._items = pd.Index(items["item_id"].astype("int64").unique())
        item_cat = pd.Series(
            items["item_category_id"].astype("int64").to_numpy(),
            index=items["item_id"].astype("int64"),
        )
        self._item_cat = item_cat
        self._cat_codes = _category_codes(categories)
        self._city = _city_codes(shops)

        # история: только месяцы до m
        g = grid.loc[grid["date_block_num"] < m, ["date_block_num", "shop_id", "item_id", "target"]]
        month = g["date_block_num"].to_numpy(dtype=np.int64)
        shop = g["shop_id"].to_numpy(dtype=np.int64)
        item = g["item_id"].to_numpy(dtype=np.int64)
        target = g["target"].to_numpy(dtype=np.float64)
        pair = _key(shop, item)

        # target пары за месяц m-k
        self._lags: dict[int, pd.Series[float]] = {
            k: _series(pair[month == m - k], target[month == m - k]) for k in TARGET_LAGS
        }

        # средний target уровня за m-1 (сетка месяца m-1 вместе с нулями)
        last = month == m - 1
        cat = _lookup(item_cat.astype("float64"), item[last]).astype(np.int64)
        city = _lookup(self._city.astype("float64"), shop[last]).astype(np.int64)
        level_keys: dict[str, Ints] = {
            "item": item[last],
            "shop": shop[last],
            "cat": cat,
            "shop_cat": _key(shop[last], cat),
            "city_item": _key(city, item[last]),
        }
        self._means: dict[str, pd.Series[float]] = {
            name: pd.Series(target[last]).groupby(keys).mean() for name, keys in level_keys.items()
        }

        # первая и последняя продажа пары, первое появление товара в сетке
        sold = target > 0
        sold_month = pd.Series(month[sold])
        self._pair_first = sold_month.groupby(pair[sold]).min().astype("float64")
        self._pair_last = sold_month.groupby(pair[sold]).max().astype("float64")
        hist = pd.DataFrame({"item": item, "month": month.astype(np.float64)})
        self._item_first: pd.Series[float] = hist.groupby("item")["month"].min()

        # цены: медиана товара по месяцам, затем последняя и средняя за историю
        s = sales.loc[
            (sales["item_cnt_day"] > 0) & (sales["date_block_num"] < m),
            ["date_block_num", "item_id", "item_price"],
        ].astype({"date_block_num": "int64", "item_id": "int64", "item_price": "float64"})
        monthly = (
            s.groupby(["item_id", "date_block_num"])["item_price"].median().reset_index()
        )  # отсортировано по item_id, date_block_num
        by_item = monthly.groupby("item_id")["item_price"]
        self._price_last: pd.Series[float] = by_item.last()
        self._price_mean: pd.Series[float] = by_item.sum() / by_item.count()
        prev = s.loc[s["date_block_num"] == m - 1]
        prev_cat = _lookup(item_cat.astype("float64"), prev["item_id"].to_numpy(dtype=np.int64))
        self._cat_price: pd.Series[float] = prev["item_price"].groupby(prev_cat).mean()

    def is_known(self, shop_id: int, item_id: int) -> bool:
        """True, если магазин и товар есть в справочниках."""
        return shop_id in self._shops and item_id in self._items

    def find_unknown(self, pairs: Sequence[tuple[int, int]]) -> tuple[list[int], list[int]]:
        """Неизвестные магазины и товары среди пар (по справочникам). Пустые списки = все известны."""
        shop, item = self._arrays(pairs)
        bad_shop = np.unique(shop[self._shops.get_indexer(pd.Index(shop)) < 0])
        bad_item = np.unique(item[self._items.get_indexer(pd.Index(item)) < 0])
        return bad_shop.tolist(), bad_item.tolist()

    def build(self, pairs: Sequence[tuple[int, int]], model_type: ModelType) -> pd.DataFrame:
        """Признаки для пар в порядке запроса (дубли допустимы), колонки = FEATURES[model_type].

        Бросает UnknownIdError со всеми неизвестными id сразу.
        """
        columns = list(FEATURES[model_type])
        bad_shops, bad_items = self.find_unknown(pairs)
        if bad_shops or bad_items:
            raise UnknownIdError(bad_shops, bad_items)
        raw_shop, item = self._arrays(pairs)
        shop = pd.Series(raw_shop).replace(SHOP_MAP).to_numpy(dtype=np.int64)
        return self._compute(shop, item)[columns]

    @staticmethod
    def _arrays(pairs: Sequence[tuple[int, int]]) -> tuple[Ints, Ints]:
        arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        return arr[:, 0], arr[:, 1]

    def _compute(self, shop: Ints, item: Ints) -> pd.DataFrame:
        m = self.target_month
        pair = _key(shop, item)
        cat = self._item_cat.reindex(item).to_numpy(dtype=np.int64)
        codes = self._cat_codes.reindex(cat)
        city = self._city.reindex(shop).to_numpy(dtype=np.int64)

        out: dict[str, NDArray[np.generic]] = {
            "shop_id": shop,
            "item_category_id": cat.astype(np.uint8),
            "category_type_id": codes["category_type_id"].to_numpy(dtype=np.uint8),
            "item_type_id": codes["item_type_id"].to_numpy(dtype=np.uint8),
            "city_id": city.astype(np.uint8),
        }

        lags = {k: _lookup(self._lags[k], pair).astype(np.float32) for k in TARGET_LAGS}
        for k, v in lags.items():
            out[f"target_lag_{k}"] = v
        filled = np.stack([np.nan_to_num(lags[k], nan=0.0) for k in range(1, 7)], axis=1)
        for w in ROLL_WINDOWS:
            out[f"roll{w}"] = filled[:, :w].astype(np.float64).mean(axis=1).astype(np.float32)
        out["diff_1_2"] = lags[1] - lags[2]

        level_keys: dict[str, Ints] = {
            "item": item,
            "shop": shop,
            "cat": cat,
            "shop_cat": _key(shop, cat),
            "city_item": _key(city, item),
        }
        for name, keys in level_keys.items():
            out[f"{name}_mean_lag_1"] = _lookup(self._means[name], keys).astype(np.float32)

        out["pair_age"] = (m - _lookup(self._pair_first, pair)).astype(np.float32)
        out["pair_since_last"] = (m - _lookup(self._pair_last, pair)).astype(np.float32)
        # товар без истории впервые появляется в месяце m
        item_first = np.fmin(_lookup(self._item_first, item), m)
        item_age = (m - item_first).astype(np.int16)
        out["item_age"] = item_age
        out["item_is_new"] = (item_age == 0).astype(np.int8)

        price_last = _lookup(self._price_last, item)
        out["item_price_last"] = price_last.astype(np.float32)
        out["item_price_ratio"] = (price_last / _lookup(self._price_mean, item)).astype(np.float32)
        out["price_rel_cat"] = (price_last / _lookup(self._cat_price, cat)).astype(np.float32)

        # календарь: date_block_num 0 = январь 2013
        n = len(shop)
        month = m % 12 + 1
        angle = 2 * np.pi * (month - 1) / 12
        days = (pd.Period("2013-01", freq="M") + m).days_in_month
        out["month"] = np.full(n, month, dtype=np.int8)
        out["month_sin"] = np.full(n, np.sin(angle), dtype=np.float32)
        out["month_cos"] = np.full(n, np.cos(angle), dtype=np.float32)
        out["days"] = np.full(n, days, dtype=np.int8)
        return pd.DataFrame(out)
