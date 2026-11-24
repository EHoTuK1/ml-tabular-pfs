"""Управление жизненным циклом модели и запусков обучения."""

import logging
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import pandas as pd

from pfs.features.spec import FEATURES, ModelType
from pfs.features.store import FeatureStore
from pfs.features.training_data import prepare_training_data, snapshot_source
from pfs.processor.predictor import LightGBMPredictor
from pfs.registry import ModelRegistry, RegisteredModel, TrainingStatus
from pfs.workflow import WorkFlowPipeline

LOGGER = logging.getLogger(__name__)


class TrainingInProgressError(RuntimeError):
    """Означает, что сервис уже выполняет обучение."""


class Processor:
    """Координирует обучение, реестр моделей и расчёт прогнозов."""

    def __init__(self, data_dir: Path, workflow: WorkFlowPipeline, registry: ModelRegistry) -> None:
        """Связывает источник данных, пайплайн обучения и реестр моделей."""
        self._data_dir = data_dir.resolve()
        self._workflow = workflow
        self._registry = registry
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = Lock()
        self._job: Future[None] | None = None
        self._predictor: LightGBMPredictor | None = None

    @property
    def model_loaded(self) -> bool:
        """Показывает готовность активной модели к прогнозу."""
        return self._predictor is not None

    def restore(self) -> None:
        """Восстанавливает модель и её признаки из активной версии реестра."""
        model = self._registry.restore_active()
        self._predictor = self._build_predictor(model) if model is not None else None

    def close(self) -> None:
        """Дожидается завершения фонового обучения при штатной остановке."""
        self._executor.shutdown()

    def predict(self, pairs: Sequence[tuple[int, int]]) -> list[float]:
        """Прогнозирует по текущей модели или вызывает RuntimeError до её загрузки."""
        predictor = self._predictor
        if predictor is None:
            raise RuntimeError("Модель ещё не загружена")
        return predictor.predict(pairs)

    def start_training(self, *, learning_rate: float, num_leaves: int, n_estimators: int) -> str:
        """Регистрирует запуск и передаёт его фоновому исполнителю.

        Если обучение уже выполняется, вызывает TrainingInProgressError.
        """
        with self._lock:
            if self._job is not None and not self._job.done():
                raise TrainingInProgressError("Обучение уже выполняется")
            run_id = self._registry.start_run(
                {
                    "learning_rate": learning_rate,
                    "num_leaves": num_leaves,
                    "n_estimators": n_estimators,
                }
            )
            try:
                self._job = self._executor.submit(
                    self._train_and_publish, run_id, learning_rate, num_leaves, n_estimators
                )
            except RuntimeError:
                self._registry.finish_failed(run_id)
                raise
        return run_id

    def get_training_status(self, run_id: str) -> TrainingStatus | None:
        """Запрашивает состояние запуска у реестра."""
        return self._registry.get_status(run_id)

    def _train_and_publish(
        self, run_id: str, learning_rate: float, num_leaves: int, n_estimators: int
    ) -> None:
        """Получает данные, передаёт их пайплайну, затем активирует версию."""
        try:
            with TemporaryDirectory(prefix="train-", dir=self._registry.root) as directory:
                work_dir = Path(directory)
                release = work_dir / "release"
                source = release / "source"
                snapshot_source(self._data_dir, source)
                data = prepare_training_data(
                    source, work_dir / "features.parquet", release / "train.csv"
                )
                result = self._workflow.train(
                    data,
                    work_dir / "result",
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    n_estimators=n_estimators,
                )
                result.metrics_file.rename(release / "metrics.json")
                version = self._registry.register(
                    run_id,
                    result.model_file,
                    release,
                    validation_rmse=result.validation_rmse,
                    validation_month=data.validation_month,
                    target_month=data.target_month,
                )
            predictor = self._build_predictor(self._registry.load_version(version))
            self._registry.activate(version)
        except Exception:
            LOGGER.exception("Ошибка запуска обучения %s", run_id)
            self._registry.finish_failed(run_id)
            return

        self._predictor = predictor
        try:
            self._registry.finish_succeeded(run_id)
        except Exception:
            LOGGER.exception("Не удалось записать завершение активированного запуска %s", run_id)

    @staticmethod
    def _build_predictor(version: RegisteredModel) -> LightGBMPredictor:
        """Создаёт предиктор из модели и истории одной версии.

        При несовпадении признаков модели со спецификацией вызывает ValueError.
        """
        if version.model.feature_name() != list(FEATURES[ModelType.LGBM]):
            raise ValueError("Признаки модели не совпадают со спецификацией")
        source = version.release / "source"
        features = FeatureStore(
            pd.read_csv(version.release / "train.csv"),
            pd.read_csv(source / "sales_train.csv"),
            pd.read_csv(source / "items.csv"),
            pd.read_csv(source / "item_categories.csv"),
            pd.read_csv(source / "shops.csv"),
            target_month=version.target_month,
        )
        return LightGBMPredictor(version.model, features)
