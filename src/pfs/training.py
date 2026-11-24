"""Точка входа для самостоятельного запуска обучения."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pfs.features.training_data import prepare_training_data
from pfs.workflow.training import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_NUM_LEAVES,
    add_training_arguments,
    train_prepared,
    validate_parameters,
)

__all__ = ["DEFAULT_LEARNING_RATE", "DEFAULT_N_ESTIMATORS", "DEFAULT_NUM_LEAVES", "main", "train"]


def train(
    data_dir: Path,
    output_dir: Path,
    *,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_leaves: int = DEFAULT_NUM_LEAVES,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
) -> dict[str, int | float]:
    """Готовит данные из CSV, обучает LightGBM и сохраняет результат в новом каталоге.

    Если каталог назначения уже существует, вызывает FileExistsError.
    """
    validate_parameters(learning_rate, num_leaves, n_estimators)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".training-", dir=output_dir.parent) as directory:
        work_dir = Path(directory)
        staging = work_dir / "result"
        staging.mkdir()
        data = prepare_training_data(data_dir, work_dir / "features.parquet", staging / "train.csv")
        metrics = train_prepared(
            data,
            staging,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            n_estimators=n_estimators,
        )
        staging.rename(output_dir)
    return metrics


def main() -> None:
    """Запускает самостоятельное обучение из командной строки."""
    parser = argparse.ArgumentParser(description="Train the PFS LightGBM model from Kaggle CSVs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/landing"))
    parser.add_argument("--output-dir", type=Path, required=True)
    add_training_arguments(parser)
    args = parser.parse_args()
    metrics = train(
        args.data_dir,
        args.output_dir,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
