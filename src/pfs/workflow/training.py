"""Обучение LightGBM на подготовленных признаках."""

import argparse
import json
from math import isfinite
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from numpy.typing import NDArray

from pfs.features.spec import CAT_FEATURES, CLIP, ModelType, TrainingData

DEFAULT_LEARNING_RATE = 0.05
DEFAULT_NUM_LEAVES = 84
DEFAULT_N_ESTIMATORS = 215


def _fit(
    features: pd.DataFrame,
    target: "pd.Series[float]",
    learning_rate: float,
    num_leaves: int,
    n_estimators: int,
) -> LGBMRegressor:
    """Обучает LightGBM с остальными параметрами из офлайн-расчёта."""
    model = LGBMRegressor(
        objective="regression",
        metric="rmse",
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        n_estimators=n_estimators,
        min_child_samples=236,
        colsample_bytree=0.6645097995051339,
        subsample=0.5756895436932168,
        subsample_freq=1,
        reg_lambda=2.8570139411689,
        cat_smooth=5.178938561126461,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    categorical_features = list(CAT_FEATURES[ModelType.LGBM])
    return model.fit(features, target, categorical_feature=categorical_features)  # pyright: ignore[reportUnknownMemberType]


def train_prepared(
    data: TrainingData,
    output_dir: Path,
    *,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_leaves: int = DEFAULT_NUM_LEAVES,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
) -> dict[str, int | float]:
    """Оценивает и сохраняет модель на переданных признаках.

    Последний месяц с таргетом отложен для RMSE. Итоговая модель обучается
    повторно с включением этого месяца. Возвращает параметры запуска и метрику.
    Неверные гиперпараметры вызывают ValueError.
    """
    validate_parameters(learning_rate, num_leaves, n_estimators)
    features = pd.read_parquet(data.feature_file, engine="pyarrow")
    target = features.pop("target")
    train_size = data.train_size
    validation_model = _fit(
        features.iloc[:train_size],
        target.iloc[:train_size],
        learning_rate,
        num_leaves,
        n_estimators,
    )
    predictions = cast(
        NDArray[np.float64], validation_model.booster_.predict(features.iloc[train_size:])
    )
    actual: NDArray[np.float64] = target.iloc[train_size:].to_numpy(dtype=np.float64)
    errors = actual - predictions.clip(*CLIP)
    rmse = float(np.mean(np.square(errors))) ** 0.5
    model = _fit(features, target, learning_rate, num_leaves, n_estimators)

    metrics = {
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "n_estimators": n_estimators,
        "validation_month": data.validation_month,
        "target_month": data.target_month,
        "validation_rmse": rmse,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(output_dir / "model.txt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def validate_parameters(learning_rate: float, num_leaves: int, n_estimators: int) -> None:
    """Проверяет диапазоны гиперпараметров и вызывает ValueError при ошибке."""
    if not isfinite(learning_rate) or learning_rate <= 0 or num_leaves < 2 or n_estimators < 1:
        raise ValueError("Требуются learning_rate > 0, num_leaves >= 2 и n_estimators >= 1")


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет общие флаги гиперпараметров."""
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--num-leaves", type=int, default=DEFAULT_NUM_LEAVES)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)


def main() -> None:
    """Запускает обучение в дочернем процессе на готовом датасете."""
    parser = argparse.ArgumentParser(description="Train LightGBM from prepared features.")
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--validation-month", type=int, required=True)
    parser.add_argument("--target-month", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_training_arguments(parser)
    args = parser.parse_args()
    metrics = train_prepared(
        TrainingData(args.feature_file, args.train_size, args.validation_month, args.target_month),
        args.output_dir,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
