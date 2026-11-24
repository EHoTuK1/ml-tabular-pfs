"""Подготовка снимка исходных данных и признаков для обучения."""

from pathlib import Path
from shutil import copyfile
from typing import cast

import pandas as pd

from pfs.features.gold import build_gold
from pfs.features.spec import ModelType, TrainingData
from pfs.features.store import FeatureStore

SOURCE_FILES = ("sales_train.csv", "items.csv", "item_categories.csv", "shops.csv")
FIRST_TRAIN_MONTH = 12


def snapshot_source(data_dir: Path, source_dir: Path) -> None:
    """Копирует CSV в снимок, общий для расчёта признаков и прогноза."""
    source_dir.mkdir(parents=True)
    for filename in SOURCE_FILES:
        copyfile(data_dir / filename, source_dir / filename)


def prepare_training_data(data_dir: Path, feature_file: Path, gold_file: Path) -> TrainingData:
    """Строит Gold и сохраняет признаки по месяцам в файл для процесса обучения.

    Каждый месяц использует только предшествующую историю. Нехватка месяцев
    для обучения и проверки вызывает ValueError.
    """
    sales = pd.read_csv(data_dir / "sales_train.csv", parse_dates=["date"], date_format="%d.%m.%Y")
    items = pd.read_csv(data_dir / "items.csv")
    categories = pd.read_csv(data_dir / "item_categories.csv")
    shops = pd.read_csv(data_dir / "shops.csv")
    gold = build_gold(sales)

    months = gold.loc[gold["date_block_num"] >= FIRST_TRAIN_MONTH]
    validation_month = int(gold["date_block_num"].max())
    if months["date_block_num"].nunique() < 2:
        raise ValueError("Для обучения и проверки нужны хотя бы два месяца начиная с месяца 12")

    feature_parts: list[pd.DataFrame] = []
    for month, rows in months.groupby("date_block_num", sort=True):
        store = FeatureStore(gold, sales, items, categories, shops, target_month=cast(int, month))
        pairs = [(int(shop), int(item)) for shop, item in rows[["shop_id", "item_id"]].to_numpy()]
        feature_parts.append(store.build(pairs, ModelType.LGBM))
    train_size = len(months) - len(feature_parts[-1])
    features = pd.concat(feature_parts, ignore_index=True)
    feature_parts.clear()
    features["target"] = months["target"].reset_index(drop=True)
    features.to_parquet(feature_file, index=False, engine="pyarrow")
    gold.to_csv(gold_file, index=False)
    return TrainingData(
        feature_file, train_size, validation_month, int(sales["date_block_num"].max()) + 1
    )
