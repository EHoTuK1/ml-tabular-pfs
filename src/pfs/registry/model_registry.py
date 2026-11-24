"""Локальный MLflow Registry и история запусков обучения."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

from lightgbm import Booster, __version__
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.lightgbm import load_model, save_model  # pyright: ignore[reportUnknownVariableType]

EXPERIMENT_NAME = "pfs-training"
MODEL_NAME = "pfs-lightgbm"
ACTIVE_ALIAS = "champion"
ARTIFACT_PATH = "release"
MODEL_PATH = f"{ARTIFACT_PATH}/model"
MODEL_VERSION_TAG = "pfs.model_version"
TARGET_MONTH_TAG = "pfs.target_month"


@dataclass(frozen=True)
class TrainingStatus:
    """Состояние запуска обучения для HTTP API."""

    run_id: str
    status: Literal["running", "succeeded", "failed"]
    model_version: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class RegisteredModel:
    """Модель и путь к снимку истории одной версии."""

    model: Booster
    release: Path
    target_month: int


class ModelRegistry:
    """Хранит версии моделей и их снимки в локальном MLflow."""

    def __init__(self, root: Path) -> None:
        """Открывает SQLite и создаёт эксперимент с локальным каталогом артефактов."""
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{(self.root / 'mlflow.db').as_posix()}"
        self._client = MlflowClient(tracking_uri=uri, registry_uri=uri)
        experiment = self._client.get_experiment_by_name(EXPERIMENT_NAME)
        self._experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else self._client.create_experiment(
                EXPERIMENT_NAME, artifact_location=(self.root / "artifacts").as_uri()
            )
        )

    def start_run(self, params: dict[str, int | float]) -> str:
        """Создаёт запуск MLflow и фиксирует его гиперпараметры."""
        run_id = self._client.create_run(self._experiment_id).info.run_id
        try:
            for name, value in params.items():
                self._client.log_param(run_id, name, str(value))
        except Exception:
            self.finish_failed(run_id)
            raise
        return run_id

    def get_status(self, run_id: str) -> TrainingStatus | None:
        """Возвращает состояние запуска этого эксперимента, если он существует."""
        try:
            run = self._client.get_run(run_id)
        except MlflowException as error:
            if error.error_code in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}:
                return None
            raise
        if run.info.experiment_id != self._experiment_id:
            return None
        if run.info.status == "FINISHED":
            return TrainingStatus(run_id, "succeeded", run.data.tags[MODEL_VERSION_TAG])
        if run.info.status in {"FAILED", "KILLED"}:
            return TrainingStatus(run_id, "failed", error="Training failed")
        return TrainingStatus(run_id, "running")

    def register(
        self,
        run_id: str,
        model_file: Path,
        release: Path,
        *,
        validation_rmse: float,
        validation_month: int,
        target_month: int,
    ) -> str:
        """Регистрирует модель со снимком и метриками, возвращает номер версии."""
        if not isfinite(validation_rmse):
            raise ValueError("RMSE на отложенном месяце должен быть конечным числом")
        model = Booster(model_file=model_file)
        save_model(model, str(release / "model"), pip_requirements=[f"lightgbm=={__version__}"])
        self._client.log_metric(run_id, "validation_rmse", validation_rmse)
        self._client.log_param(run_id, "validation_month", str(validation_month))
        self._client.log_param(run_id, "target_month", str(target_month))
        self._client.log_artifacts(run_id, str(release), ARTIFACT_PATH)
        try:
            self._client.create_registered_model(MODEL_NAME)
        except MlflowException as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
        artifact_uri = self._client.get_run(run_id).info.artifact_uri
        version = self._client.create_model_version(
            MODEL_NAME,
            f"{artifact_uri}/{MODEL_PATH}",
            run_id,
            tags={TARGET_MONTH_TAG: str(target_month)},
        )
        version_id = str(version.version)
        self._client.set_tag(run_id, MODEL_VERSION_TAG, version_id)
        return version_id

    def load_version(self, version: str) -> RegisteredModel:
        """Загружает модель и возвращает её вместе со снимком истории."""
        record = self._client.get_model_version(MODEL_NAME, version)
        if record.run_id is None:
            raise ValueError("У версии модели отсутствует запуск обучения")
        release = Path(self._client.download_artifacts(record.run_id, "")) / ARTIFACT_PATH
        model = load_model(str(release / "model"))
        if not isinstance(model, Booster):
            raise ValueError("Версия не содержит модель LightGBM Booster")
        return RegisteredModel(model, release, int(record.tags[TARGET_MONTH_TAG]))

    def restore_active(self) -> RegisteredModel | None:
        """Загружает активную версию и завершает прерванные при остановке запуски."""
        try:
            active = self._client.get_model_version_by_alias(MODEL_NAME, ACTIVE_ALIAS)
        except MlflowException as error:
            if error.error_code not in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}:
                raise
            active = None

        active_run = active.run_id if active is not None else None
        for run in self._client.search_runs([self._experiment_id], "attributes.status = 'RUNNING'"):
            status = "FINISHED" if run.info.run_id == active_run else "KILLED"
            self._client.set_terminated(run.info.run_id, status)
        return self.load_version(str(active.version)) if active is not None else None

    def activate(self, version: str) -> None:
        """Переводит алиас активной модели на проверенную версию."""
        self._client.set_registered_model_alias(MODEL_NAME, ACTIVE_ALIAS, version)

    def finish_succeeded(self, run_id: str) -> None:
        """Отмечает запуск как успешный после активации."""
        self._client.set_terminated(run_id, "FINISHED")

    def finish_failed(self, run_id: str) -> None:
        """Отмечает запуск как неудачный без изменения активной версии."""
        self._client.set_terminated(run_id, "FAILED")
