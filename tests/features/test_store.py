import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfs.errors import UnknownIdError
from pfs.features import FEATURES, FeatureStore, ModelType

from .conftest import Raw, load_tmp_script, offline_features

TARGET_MONTH = 15


def make_store(grid: pd.DataFrame, raw: Raw, target_month: int = TARGET_MONTH) -> FeatureStore:
    return FeatureStore(
        grid, raw.sales, raw.items, raw.categories, raw.shops, target_month=target_month
    )


def all_pairs(raw: Raw) -> list[tuple[int, int]]:
    """Все магазины x товары, кроме дублей 0/1/11 (после переобозначения дали бы повторы)."""
    shops = [s for s in raw.shops["shop_id"] if s not in (0, 1, 11)]
    return [(int(s), int(i)) for s in shops for i in raw.items["item_id"]]


def assert_parity(online: pd.DataFrame, offline: pd.DataFrame) -> None:
    cols = list(FEATURES[ModelType.LGBM])
    pd.testing.assert_frame_equal(
        online.reset_index(drop=True),
        offline[cols].reset_index(drop=True),
        check_dtype=False,
        rtol=1e-5,
    )


@pytest.mark.parametrize("target_month", [TARGET_MONTH, 13, 3])
def test_parity_with_offline(grid: pd.DataFrame, raw: Raw, target_month: int) -> None:
    """Главный тест: признаки онлайн = признаки офлайн-пайплайна из tmp/scripts.

    Пары: с историей, известные id без продаж (товары 30..39, магазин 30), разные месяцы,
    в т.ч. 3 (лаг 12 ещё пуст).
    """
    pairs = all_pairs(raw)
    online = make_store(grid, raw, target_month).build(pairs, ModelType.LGBM)
    offline = offline_features(grid, raw, pairs, target_month)
    assert_parity(online, offline)
    # данные не вырожденные: есть и пустые, и заполненные лаги
    assert online["target_lag_1"].isna().any()
    assert online["target_lag_1"].notna().any()


def test_dtypes_match_offline(grid: pd.DataFrame, raw: Raw) -> None:
    pairs = all_pairs(raw)
    online = make_store(grid, raw).build(pairs, ModelType.LGBM)
    offline = offline_features(grid, raw, pairs, TARGET_MONTH)
    cols = list(FEATURES[ModelType.LGBM])
    assert online.dtypes.to_dict() == offline[cols].dtypes.to_dict()


def test_duplicate_shops_mapped(grid: pd.DataFrame, raw: Raw) -> None:
    """Магазин 0 — дубль 57: признаки как у 57, shop_id в признаках тоже 57."""
    store = make_store(grid, raw)
    df = store.build([(0, 5), (57, 5)], ModelType.LGBM)
    pd.testing.assert_series_equal(df.iloc[0], df.iloc[1], check_names=False)
    assert df["shop_id"].tolist() == [57, 57]


def test_columns_order_and_rows(grid: pd.DataFrame, raw: Raw) -> None:
    store = make_store(grid, raw)
    pairs = [(2, 3), (10, 7), (2, 3), (30, 35)]
    df = store.build(pairs, ModelType.LGBM)
    assert list(df.columns) == list(FEATURES[ModelType.LGBM])
    assert len(df) == len(pairs)
    assert isinstance(df.index, pd.RangeIndex)
    pd.testing.assert_series_equal(df.iloc[0], df.iloc[2], check_names=False)
    single = store.build([(10, 7)], ModelType.LGBM)
    pd.testing.assert_frame_equal(single, df.iloc[[1]].reset_index(drop=True))


def test_cold_start_known_ids(grid: pd.DataFrame, raw: Raw) -> None:
    """Магазин 30 без продаж, товар 35 никогда не продавался: лаги пустые, ошибки нет."""
    row = make_store(grid, raw).build([(30, 35)], ModelType.LGBM).iloc[0]
    lag_cols = [c for c in row.index if c.startswith("target_lag_")]
    assert row[lag_cols].isna().all()
    assert np.isnan(row["pair_age"])
    assert row["item_age"] == 0
    assert row["item_is_new"] == 1
    assert row["roll3"] == 0


def test_unknown_ids_all_reported(grid: pd.DataFrame, raw: Raw) -> None:
    store = make_store(grid, raw)
    pairs = [(2, 3), (99, 3), (2, 777), (98, 776), (99, 3)]
    assert store.find_unknown(pairs) == ([98, 99], [776, 777])
    with pytest.raises(UnknownIdError) as exc:
        store.build(pairs, ModelType.LGBM)
    assert exc.value.shop_ids == [98, 99]
    assert exc.value.item_ids == [776, 777]


def test_find_unknown_all_known(grid: pd.DataFrame, raw: Raw) -> None:
    assert make_store(grid, raw).find_unknown([(2, 3), (0, 39)]) == ([], [])


def test_empty_request(grid: pd.DataFrame, raw: Raw) -> None:
    df = make_store(grid, raw).build([], ModelType.LGBM)
    assert df.empty
    assert list(df.columns) == list(FEATURES[ModelType.LGBM])


def test_future_history_ignored(grid: pd.DataFrame, raw: Raw) -> None:
    """Месяцы >= target_month в данных не влияют на признаки (нет утечки)."""
    pairs = all_pairs(raw)[:50]
    cut_grid = grid[grid["date_block_num"] < TARGET_MONTH]
    cut_raw = Raw(
        sales=raw.sales[raw.sales["date_block_num"] < TARGET_MONTH],
        items=raw.items,
        categories=raw.categories,
        shops=raw.shops,
    )
    full = make_store(grid, raw).build(pairs, ModelType.LGBM)
    cut = make_store(cut_grid, cut_raw).build(pairs, ModelType.LGBM)
    pd.testing.assert_frame_equal(full, cut)


# Паритет на реальных данных курсовой: долго (полный офлайн-пайплайн на ~11 млн строк),
# поэтому запускается только с PFS_DATA_DIR=<папка с исходными CSV Kaggle (bronze)>.
# Сетку (gold) строит prepare_data из tmp, пока её не создаёт WorkFlowPipeline.
DATA_DIR = os.environ.get("PFS_DATA_DIR")


@pytest.mark.skipif(DATA_DIR is None, reason="PFS_DATA_DIR не задан")
def test_parity_real_data() -> None:
    assert DATA_DIR is not None
    d = Path(DATA_DIR)
    month = 33
    grid = load_tmp_script("prepare_data").prepare_raw_data(d / "sales_train.csv")
    raw = Raw(
        sales=pd.read_csv(d / "sales_train.csv"),
        items=pd.read_csv(d / "items.csv"),
        categories=pd.read_csv(d / "item_categories.csv"),
        shops=pd.read_csv(d / "shops.csv"),
    )
    real = grid.loc[grid["date_block_num"] == month, ["shop_id", "item_id"]]
    sample = real.sample(5000, random_state=0)
    pairs = [(int(s), int(i)) for s, i in sample.itertuples(index=False)]
    online = make_store(grid, raw, month).build(pairs, ModelType.LGBM)
    offline = offline_features(grid, raw, pairs, month)
    assert_parity(online, offline)

 
def test_is_known(grid: pd.DataFrame, raw: Raw) -> None:
    store = make_store(grid, raw)
    assert store.is_known(2, 3)
    assert store.is_known(30, 39)  # без продаж, но в справочниках
    assert not store.is_known(99, 3)
    assert not store.is_known(2, 777)
