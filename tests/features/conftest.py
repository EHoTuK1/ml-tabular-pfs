"""Синтетические данные в формате Kaggle и офлайн-расчёт признаков из tmp/scripts."""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
TMP_SCRIPTS = ROOT / "tmp" / "scripts"


def load_tmp_script(name: str) -> Any:
    """Модуль из tmp/scripts (справочный офлайн-код) без правки sys.path."""
    spec = importlib.util.spec_from_file_location(f"tmp_{name}", TMP_SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class Raw:
    sales: pd.DataFrame
    items: pd.DataFrame
    categories: pd.DataFrame
    shops: pd.DataFrame


def make_raw(n_months: int = 16, seed: int = 0) -> Raw:
    """Небольшой набор, похожий на Kaggle: дубли магазинов, возвраты, товары без продаж."""
    rng = np.random.default_rng(seed)
    categories = pd.DataFrame(
        {
            "item_category_id": [0, 1, 2, 3, 4],
            "item_category_name": [
                "Игры - PS3",
                "Игры - XBOX",
                "Кино - DVD",
                "Книги - Аудио",
                "Подарки",  # без '-' -> item_type_id = 0
            ],
        }
    )
    # 0, 1, 11 — дубли 57, 58, 10; 30 — магазин без продаж
    shops = pd.DataFrame(
        {
            "shop_id": [0, 1, 2, 10, 11, 12, 30, 57, 58],
            "shop_name": [
                "!Якутск Орджоникидзе, 56 фран",
                "!Якутск ТЦ Центральный фран",
                "Адыгея ТЦ Мега",
                "Жуковский ул. Чкалова 39м?",
                "Жуковский ул. Чкалова 39м²",
                "Интернет-магазин ЧС",
                "Москва ТЦ Новый",
                "Якутск Орджоникидзе, 56",
                "Якутск ТЦ Центральный",
            ],
        }
    )
    n_items = 40  # товары 30..39 не продаются никогда
    items = pd.DataFrame(
        {
            "item_name": [f"item {i}" for i in range(n_items)],
            "item_id": np.arange(n_items),
            "item_category_id": np.arange(n_items) % 5,
        }
    )
    selling_shops = np.array([0, 1, 2, 10, 11, 12, 57, 58])
    # редкие продажи и товары, выходящие в разные месяцы: у пар появляются пропуски в истории
    n = 1500
    item_id = rng.integers(0, 30, n)
    month = rng.integers(item_id // 3, n_months)
    sales = pd.DataFrame(
        {
            "date": pd.Timestamp("2013-01-01")
            + pd.to_timedelta(month * 31 + rng.integers(0, 28, n), unit="D"),
            "date_block_num": month,
            "shop_id": rng.choice(selling_shops, n),
            "item_id": item_id,
            "item_price": rng.choice([99.0, 149.0, 299.0, 999.0], n) * rng.uniform(0.8, 1.2, n),
            "item_cnt_day": rng.choice([-1, 1, 1, 1, 2, 3, 25], n),
        }
    )
    return Raw(sales=sales, items=items, categories=categories, shops=shops)


def make_grid(raw: Raw) -> pd.DataFrame:
    """Сетка как в prepare_data.prepare_raw_data (без чтения CSV)."""
    prep = load_tmp_script("prepare_data")
    sales = prep.fix_nonpositive_prices(raw.sales)
    sales = prep.drop_duplicate_transactions(sales)
    sales = prep.merge_duplicate_shops(sales)
    return prep.add_monthly_target(prep.build_grid(sales), sales)


def offline_features(
    grid: pd.DataFrame, raw: Raw, pairs: list[tuple[int, int]], target_month: int
) -> pd.DataFrame:
    """Признаки месяца target_month по ноутбуку submit.ipynb: пары дописаны как тест-месяц."""
    fe = load_tmp_script("feature_engineering")
    prep = load_tmp_script("prepare_data")
    history = grid[grid["date_block_num"] < target_month]
    test = pd.DataFrame(pairs, columns=["shop_id", "item_id"])
    test = prep.merge_duplicate_shops(test).assign(date_block_num=target_month, ID=range(len(test)))
    df = pd.concat([history, test[["ID", "date_block_num", "shop_id", "item_id"]]])
    df = df.astype({"date_block_num": "int64", "shop_id": "int64", "item_id": "int64"})
    df = df.reset_index(drop=True)

    df = fe.add_categories(df, raw.categories, raw.items)
    df = fe.add_cities(df, raw.shops)
    df = fe.add_target_lags(df, (1, 2, 3, 4, 5, 6, 12))
    df = fe.add_lag_stats(df)
    df = fe.add_mean_lags(df)
    df = fe.add_age_features(df)
    df = fe.add_price_features(df, raw.sales, raw.items)
    df = fe.add_calendar(df)
    out = df[df["date_block_num"] == target_month].sort_values("ID")
    return out.reset_index(drop=True)


@pytest.fixture(scope="session")
def raw() -> Raw:
    return make_raw()


@pytest.fixture(scope="session")
def grid(raw: Raw) -> pd.DataFrame:
    return make_grid(raw)
