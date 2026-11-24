"""Сборка месячной сетки Gold из транзакций Kaggle."""

import pandas as pd

from pfs.features.spec import CLIP

GOLD_KEYS = ["date_block_num", "shop_id", "item_id"]
SHOP_MAP: dict[int, int] = {0: 57, 1: 58, 11: 10}


def build_gold(sales: pd.DataFrame) -> pd.DataFrame:
    """Собирает сетку и таргет по дневным транзакциям с разобранным полем date.

    Для каждого месяца в сетку попадают все сочетания магазинов и товаров,
    у которых были положительные продажи. Таргет учитывает возвраты, заполняет
    отсутствующие продажи нулём и обрезается до диапазона CLIP.
    """
    transactions = (
        sales[["date", *GOLD_KEYS, "item_cnt_day"]]
        .sort_values("date", ignore_index=True)
        .astype({column: "int16" for column in [*GOLD_KEYS, "item_cnt_day"]})
        .drop_duplicates(subset=["date", "shop_id", "item_id"], ignore_index=True)
    )
    transactions["shop_id"] = transactions["shop_id"].replace(SHOP_MAP)
    positive = transactions.loc[transactions["item_cnt_day"] > 0, GOLD_KEYS]
    parts: list[pd.DataFrame] = []
    for month, group in positive.groupby("date_block_num"):
        grid = pd.MultiIndex.from_product(
            [[month], group["shop_id"].unique(), group["item_id"].unique()],
            names=GOLD_KEYS,
        )
        parts.append(grid.to_frame(index=False))
    if not parts:
        raise ValueError("Sales must contain at least one positive transaction")

    grid = pd.concat(parts, ignore_index=True)
    totals = transactions.groupby(GOLD_KEYS, as_index=False)["item_cnt_day"].sum()
    gold = grid.merge(totals, on=GOLD_KEYS, how="left", validate="one_to_one")
    gold["target"] = gold.pop("item_cnt_day").fillna(0).clip(*CLIP).astype("float32")
    return gold.sort_values(GOLD_KEYS, ignore_index=True)
