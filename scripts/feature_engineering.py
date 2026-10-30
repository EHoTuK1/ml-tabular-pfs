"""Признаки для месячной сетки (date_block_num, shop_id, item_id, target).

Каждая функция добавляет колонки и не меняет число строк. Все признаки строятся
только по месяцам строго до m, иначе модель видит свой target (утечка).

Порядок вызова (более поздние функции используют колонки более ранних):
    df = add_categories(df, df_cat, df_items)
    df = add_cities(df, df_shops)
    df = add_target_lags(df, (1, 2, 3, 4, 5, 6, 12))
    df = add_lag_stats(df)
    df = add_mean_lags(df)
    df = add_age_features(df)
    df = add_price_features(df, sales, df_items)
    df = add_calendar(df)
"""

import numpy as np
import pandas as pd

KEYS = ["date_block_num", "shop_id", "item_id"]
PAIR = ["shop_id", "item_id"]

# имя уровня -> ключи группировки для add_mean_lags
MEAN_LEVELS: dict[str, tuple[str, ...]] = {
    "item": ("item_id",),
    "shop": ("shop_id",),
    "cat": ("item_category_id",),
    "shop_cat": ("shop_id", "item_category_id"),
    "city_item": ("city_id", "item_id"),
}


def add_categories(
    df_train: pd.DataFrame, df_categories: pd.DataFrame, df_items: pd.DataFrame
) -> pd.DataFrame:
    """Категория товара и её части до и после '-': category_type_id, item_type_id."""
    parts = df_categories["item_category_name"].str.split("-", n=1, expand=True)
    df_categories = df_categories.assign(
        category_type_id=parts[0].str.strip().rank(method="dense").astype("uint8"),
        item_type_id=parts[1].str.strip().rank(method="dense").fillna(0).astype("uint8"),
    )

    df_train = df_train.merge(
        df_items[["item_id", "item_category_id"]], on="item_id", how="left", validate="many_to_one"
    )
    df_train = df_train.merge(
        df_categories[["item_category_id", "category_type_id", "item_type_id"]],
        on="item_category_id",
        how="left",
        validate="many_to_one",
    )
    df_train["item_category_id"] = df_train["item_category_id"].astype("uint8")
    return df_train


def add_cities(df_train: pd.DataFrame, df_shops: pd.DataFrame) -> pd.DataFrame:
    """city_id по первому слову названия магазина; интернет, склад и выездная торговля -> 'Прочее'."""
    other = df_shops["shop_name"].str.contains(
        "Интернет-магазин|Цифровой склад|Выездная Торговля", na=False, case=False
    )
    city = (
        df_shops["shop_name"]
        .str.replace("!", "", regex=False)
        .str.strip()
        .mask(other, "Прочее")
        .str.split(n=1)
        .str[0]
    )
    df_shops = df_shops.assign(city_id=city.rank(method="dense").fillna(0).astype("uint8"))
    return df_train.merge(
        df_shops[["shop_id", "city_id"]], on="shop_id", how="left", validate="many_to_one"
    )


def add_target_lags(
    df_train: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3, 6, 12)
) -> pd.DataFrame:
    """target пары за месяц m-k; NaN = пары не было в сетке, 0 = была, но не продалась."""
    # merge, а не shift: пара есть в сетке не каждый месяц
    src = df_train[[*KEYS, "target"]]
    for k in lags:
        col = f"target_lag_{k}"
        lagged = src.assign(date_block_num=src["date_block_num"] + k).rename(
            columns={"target": col}
        )
        df_train = df_train.merge(lagged, on=KEYS, how="left", validate="one_to_one")
        df_train[col] = df_train[col].astype("float32")
    return df_train


def add_lag_stats(df_train: pd.DataFrame, windows: tuple[int, ...] = (3, 6)) -> pd.DataFrame:
    """roll{w}: среднее лагов 1..w (NaN -> 0) и diff_1_2 = lag_1 - lag_2."""
    need = [f"target_lag_{k}" for k in range(1, max(windows) + 1)]
    missing = [c for c in need if c not in df_train.columns]
    if missing:
        raise KeyError(f"сначала add_target_lags, нет колонок: {missing}")

    # NaN лага = пары не было в сетке = не продавалась
    lags = df_train[need].fillna(0)
    rolls = {f"roll{w}": lags.iloc[:, :w].mean(axis=1).astype("float32") for w in windows}
    return df_train.assign(
        **rolls,
        diff_1_2=(df_train["target_lag_1"] - df_train["target_lag_2"]).astype("float32"),
    )


def add_mean_lags(
    df_train: pd.DataFrame,
    levels: dict[str, tuple[str, ...]] = MEAN_LEVELS,
    lags: tuple[int, ...] = (1,),
) -> pd.DataFrame:
    """{name}_mean_lag_{k}: средний target уровня в месяце m-k; NaN = уровня не было в сетке."""
    for name, keys in levels.items():
        group = ["date_block_num", *keys]
        means = df_train.groupby(group, as_index=False)["target"].mean()
        for k in lags:
            col = f"{name}_mean_lag_{k}"
            lagged = means.assign(date_block_num=means["date_block_num"] + k).rename(
                columns={"target": col}
            )
            df_train = df_train.merge(lagged, on=group, how="left", validate="many_to_one")
            df_train[col] = df_train[col].astype("float32")
    return df_train


def add_age_features(df_train: pd.DataFrame) -> pd.DataFrame:
    """Возраст товара (item_age, item_is_new) и пары (pair_age, pair_since_last) по продажам до m."""
    full = df_train[[*KEYS, "target"]].sort_values([*PAIR, "date_block_num"])
    full["sale_month"] = full["date_block_num"].where(full["target"] > 0)
    full["first"] = full.groupby(PAIR)["sale_month"].cummin()
    full["last"] = full["sale_month"]
    full[["first", "last"]] = full.groupby(PAIR)[["first", "last"]].ffill()
    prev = full.groupby(PAIR)[["first", "last"]].shift(1)  # без сдвига строка видит свой target

    # товар есть в сетке месяца, только если в нём продался: первое появление = первая продажа
    item_first = df_train.groupby("item_id")["date_block_num"].transform("min")
    item_age = (df_train["date_block_num"] - item_first).astype("int16")

    # assign выравнивает full и prev по индексу, порядок строк df_train сохраняется
    return df_train.assign(
        pair_age=(full["date_block_num"] - prev["first"]).astype("float32"),
        pair_since_last=(full["date_block_num"] - prev["last"]).astype("float32"),
        item_age=item_age,
        item_is_new=(item_age == 0).astype("int8"),
    )


def add_price_features(
    df_train: pd.DataFrame, sales: pd.DataFrame, df_items: pd.DataFrame
) -> pd.DataFrame:
    """Цена товара до m: item_price_last, item_price_ratio, price_rel_cat (sales — очищенные транзакции)."""
    # цену месяца m не берём: она есть только у проданных в m товаров
    sold = sales.loc[sales["item_cnt_day"] > 0, ["date_block_num", "item_id", "item_price"]]
    sold = sold.astype({"date_block_num": "int64", "item_id": "int64"})

    monthly = (
        sold.groupby(["date_block_num", "item_id"], as_index=False)
        .agg(item_price_last=("item_price", "median"))
        .sort_values(["item_id", "date_block_num"])
    )
    g = monthly.groupby("item_id")["item_price_last"]
    monthly["item_price_mean"] = g.cumsum() / (g.cumcount() + 1)

    # последняя строка monthly с месяцем строго < m; merge_asof требует общий тип ключей
    left = df_train[["date_block_num", "item_id"]].astype("int64").reset_index()
    priced = (
        pd.merge_asof(
            left.sort_values("date_block_num"),
            monthly.sort_values("date_block_num"),
            on="date_block_num",
            by="item_id",
            allow_exact_matches=False,
        )
        .set_index("index")
        .reindex(df_train.index)
    )

    cat_price = (
        sold.merge(df_items[["item_id", "item_category_id"]], on="item_id", how="left")
        .groupby(["date_block_num", "item_category_id"], as_index=False)
        .agg(cat_price_lag_1=("item_price", "mean"))
    )
    cat_price["date_block_num"] += 1
    cat_lag = (
        df_train[["date_block_num", "item_category_id"]]
        .astype("int64")
        .merge(
            cat_price, on=["date_block_num", "item_category_id"], how="left", validate="many_to_one"
        )["cat_price_lag_1"]
        .to_numpy()
    )

    last = priced["item_price_last"]
    return df_train.assign(
        item_price_last=last.astype("float32"),
        item_price_ratio=(last / priced["item_price_mean"]).astype("float32"),
        price_rel_cat=(last.to_numpy() / cat_lag).astype("float32"),
    )


def add_calendar(df_train: pd.DataFrame) -> pd.DataFrame:
    """month, month_sin/cos (месяц на окружности) и days; date_block_num 0 = январь 2013."""
    n_blocks = int(df_train["date_block_num"].max()) + 1
    blocks = pd.DataFrame({"date_block_num": np.arange(n_blocks)})
    month = blocks["date_block_num"] % 12 + 1
    angle = 2 * np.pi * (month - 1) / 12
    period = pd.period_range("2013-01", periods=n_blocks, freq="M")
    blocks = blocks.assign(
        date_block_num=blocks["date_block_num"].astype(df_train["date_block_num"].dtype),
        month=month.astype("int8"),
        month_sin=np.sin(angle).astype("float32"),
        month_cos=np.cos(angle).astype("float32"),
        days=np.asarray(period.days_in_month, dtype="int8"),
    )
    return df_train.merge(blocks, on="date_block_num", how="left", validate="many_to_one")
