"""Обучение LightGBM в отдельном процессе на переданном наборе признаков."""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from pfs.features.spec import TrainingData


@dataclass(frozen=True)
class TrainingResult:
    """Артефакты и метрики завершившегося обучения."""

    model_file: Path
    metrics_file: Path
    validation_rmse: float


class WorkFlowPipeline:
    """Принимает подготовленный набор данных и обучает модель."""

    def train(
        self,
        data: TrainingData,
        output_dir: Path,
        *,
        learning_rate: float,
        num_leaves: int,
        n_estimators: int,
    ) -> TrainingResult:
        """Запускает обучение и возвращает артефакты из output_dir.

        При ошибке процесса обучения пробрасывает CalledProcessError.
        """
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pfs.workflow.training",
                "--feature-file",
                str(data.feature_file),
                "--train-size",
                str(data.train_size),
                "--validation-month",
                str(data.validation_month),
                "--target-month",
                str(data.target_month),
                "--output-dir",
                str(output_dir),
                "--learning-rate",
                str(learning_rate),
                "--num-leaves",
                str(num_leaves),
                "--n-estimators",
                str(n_estimators),
            ],
            check=True,
            stdin=subprocess.DEVNULL,
        )
        metrics_file = output_dir / "metrics.json"
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        return TrainingResult(
            model_file=output_dir / "model.txt",
            metrics_file=metrics_file,
            validation_rmse=float(metrics["validation_rmse"]),
        )
