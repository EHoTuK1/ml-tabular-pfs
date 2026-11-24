"""Реестр моделей: какие признаки и в каком порядке ждёт каждая модель.

Порядок колонок важен: модель обучена на DataFrame с ровно этими колонками.
Новая модель = новый член ModelType и запись в FEATURES (и CAT_FEATURES).
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ModelType(StrEnum):
    """Поддерживаемые модели и их наборы признаков."""

    LGBM = "lgbm"


@dataclass(frozen=True)
class TrainingData:
    """Подготовленные признаки и таргет с границей отложенного месяца."""

    feature_file: Path
    train_size: int
    validation_month: int
    target_month: int


CLIP = (0.0, 20.0)

# Совпадает с FEATURES из tmp/notebooks/3_models/lgbm (model.ipynb, submit.ipynb).
_LGBM_FEATURES = (
    "shop_id",
    "item_category_id",
    "category_type_id",
    "item_type_id",
    "city_id",
    "target_lag_1",
    "target_lag_2",
    "target_lag_3",
    "target_lag_4",
    "target_lag_5",
    "target_lag_6",
    "target_lag_12",
    "roll3",
    "roll6",
    "diff_1_2",
    "item_mean_lag_1",
    "shop_mean_lag_1",
    "cat_mean_lag_1",
    "shop_cat_mean_lag_1",
    "city_item_mean_lag_1",
    "pair_age",
    "pair_since_last",
    "item_age",
    "item_is_new",
    "item_price_last",
    "item_price_ratio",
    "price_rel_cat",
    "month",
    "month_sin",
    "month_cos",
    "days",
)

FEATURES: dict[ModelType, tuple[str, ...]] = {ModelType.LGBM: _LGBM_FEATURES}

CAT_FEATURES: dict[ModelType, tuple[str, ...]] = {
    ModelType.LGBM: ("shop_id", "item_category_id", "category_type_id", "item_type_id", "city_id"),
}
