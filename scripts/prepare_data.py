"""Чистка транзакций и сборка месячной сетки"""

from pathlib import Path

import pandas as pd

KEYS = ["date_block_num", "shop_id", "item_id"]
SHOP_MAP = {0: 57, 1: 58, 11: 10}


def load_sales(path: str | Path) -> pd.DataFrame:
    """Читает sales_train.csv, сортирует по дате и уменьшает типы."""
    df = pd.read_csv(path, parse_dates=["date"], date_format="%d.%m.%Y").sort_values(
        by="date", ignore_index=True
    )
    return df.astype(
        {
            "date_block_num": "int16",
            "shop_id": "int16",
            "item_id": "int16",
            "item_price": "float32",
            "item_cnt_day": "int16",
        }
    )


def fix_nonpositive_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Цена <= 0 (ошибка записи) -> медиана положительных цен этого товара."""
    valid = df["item_price"].where(df["item_price"] > 0)
    median = valid.groupby(df["item_id"]).transform("median")
    return df.assign(item_price=valid.fillna(median))


def drop_duplicate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет одну транзакцию на (день, магазин, товар)."""
    return df.drop_duplicates(subset=["date", "shop_id", "item_id"], ignore_index=True)


def mark_refunds(df: pd.DataFrame) -> pd.DataFrame:
    """Флаг возврата: отрицательное количество за день."""
    return df.assign(is_refund=df["item_cnt_day"] < 0)


def merge_duplicate_shops(df: pd.DataFrame, shop_map: dict[int, int] = SHOP_MAP) -> pd.DataFrame:
    """Переобозначает магазины-дубли в сторону ID, которые есть в тесте."""
    return df.assign(shop_id=df["shop_id"].replace(shop_map))


def build_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Для каждого месяца: магазины с продажами x товары с продажами (декартово произведение)."""
    pos = df[df["item_cnt_day"] > 0]
    parts = [
        pd.MultiIndex.from_product(  # type: ignore
            [[m], g["shop_id"].unique(), g["item_id"].unique()], names=KEYS
        ).to_frame(index=False)
        for m, g in pos.groupby("date_block_num")
    ]
    return pd.concat(parts, ignore_index=True)  # type: ignore


def add_monthly_target(grid: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Сумма продаж пары за месяц (с возвратами), пропуски -> 0, клиппинг в [0, 20]."""
    agg = df.groupby(KEYS, as_index=False)["item_cnt_day"].sum()
    out = grid.merge(agg, on=KEYS, how="left")
    out["target"] = out.pop("item_cnt_day").fillna(0).clip(0, 20).astype('float32')
    return out.sort_values(KEYS, ignore_index=True)


def prepare_raw_data(train_path: str | Path) -> pd.DataFrame:
    """sales_train.csv -> длинная таблица (date_block_num, shop_id, item_id, target)."""
    sales = load_sales(train_path)
    sales = fix_nonpositive_prices(sales)
    sales = drop_duplicate_transactions(sales)
    sales = mark_refunds(sales)
    sales = merge_duplicate_shops(sales)
    return add_monthly_target(build_grid(sales), sales)


def append_test_month(df: pd.DataFrame, test_path: str | Path, month: int = 34) -> pd.DataFrame:
    """Добавляет пары из test.csv как месяц 34 с target = NaN; ID остаётся для сабмита."""
    test = pd.read_csv(test_path)
    test = merge_duplicate_shops(test).assign(date_block_num=month)
    test = test.astype({"date_block_num": "int16", "shop_id": "int16", "item_id": "int16"})
    return pd.concat([df, test[["ID", *KEYS]]], ignore_index=True)


if __name__ == "__main__":
    df = prepare_raw_data("data/landing/sales_train.csv")
    df.info()
