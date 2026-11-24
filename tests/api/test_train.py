"""Проверяет HTTP-обучение с настоящим локальным MLflow."""

from pathlib import Path
from time import monotonic, sleep
from typing import cast

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from lightgbm import Booster
from mlflow import MlflowClient
from numpy.typing import NDArray

from pfs.features.spec import FEATURES, ModelType
from pfs.registry import TrainingStatus
from pfs.service import create_service

from ..features.conftest import make_raw, offline_features


def wait_for_training(client: TestClient, run_id: str) -> TrainingStatus:
    """Ожидает терминального состояния с ограничением времени."""
    deadline = monotonic() + 60
    while monotonic() < deadline:
        response = client.get(f"/train/{run_id}")
        assert response.status_code == 200
        result = TrainingStatus(**response.json())
        if result.status != "running":
            return result
        sleep(0.1)
    raise AssertionError(f"Обучение {run_id} не завершилось за 60 секунд")


def test_training_publishes_snapshot_and_preserves_active_model_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Проверяет контракт запуска, прогноз и восстановление из реестра."""
    data_dir = tmp_path / "source"
    data_dir.mkdir()
    raw = make_raw()
    raw.sales.to_csv(data_dir / "sales_train.csv", index=False, date_format="%d.%m.%Y")
    raw.items.to_csv(data_dir / "items.csv", index=False)
    raw.categories.to_csv(data_dir / "item_categories.csv", index=False)
    raw.shops.to_csv(data_dir / "shops.csv", index=False)
    state_dir = tmp_path / "service"
    monkeypatch.setenv("PFS_DATA_DIR", str(data_dir))
    monkeypatch.setenv("PFS_STATE_DIR", str(state_dir))
    pairs = [(2, 35), (2, 3), (57, 12), (2, 35)]
    request = {"items": [{"shop_id": shop, "item_id": item} for shop, item in pairs]}

    with TestClient(create_service()) as client:
        assert client.get("/health").json()["model_loaded"] is False
        assert client.post("/predict", json=request).status_code == 503
        assert client.get("/train/unknown").status_code == 404
        assert client.post("/train", json={"num_leaves": 1}).status_code == 422
        accepted = client.post("/train", json={"n_estimators": 2})
        assert accepted.status_code == 202
        run_id = accepted.json()["run_id"]
        assert client.post("/train", json={}).status_code == 409

        result = wait_for_training(client, run_id)
        assert result.status == "succeeded"
        assert result.model_version == "1"
        assert client.get("/health").json()["model_loaded"] is True
        response = client.post("/predict", json=request)
        assert response.status_code == 200
        forecast = response.json()
        assert [(row["shop_id"], row["item_id"]) for row in forecast["predictions"]] == pairs
        tracking = MlflowClient(tracking_uri=f"sqlite:///{(state_dir / 'mlflow.db').as_posix()}")
        release = Path(tracking.download_artifacts(run_id, "")) / "release"
        grid = pd.read_csv(release / "train.csv")
        unique_pairs = list(dict.fromkeys(pairs))
        features = offline_features(grid, raw, unique_pairs, target_month=16)
        model = Booster(model_file=release / "model" / "model.lgb")
        expected = cast(
            NDArray[np.float64], model.predict(features[list(FEATURES[ModelType.LGBM])])
        ).clip(0, 20)
        np.testing.assert_allclose(
            [row["item_cnt_month"] for row in forecast["predictions"]],
            expected[[0, 1, 2, 0]],
        )

        accepted = client.post("/train", json={"learning_rate": 0.1, "n_estimators": 1})
        assert accepted.status_code == 202
        next_run_id = accepted.json()["run_id"]
        assert wait_for_training(client, next_run_id).model_version == "2"
        champion = tracking.get_model_version_by_alias("pfs-lightgbm", "champion")
        assert str(champion.version) == "2"
        forecast = client.post("/predict", json=request).json()

        (data_dir / "sales_train.csv").rename(data_dir / "moved.csv")
        accepted = client.post("/train", json={})
        assert accepted.status_code == 202
        assert wait_for_training(client, accepted.json()["run_id"]).status == "failed"
        assert client.post("/predict", json=request).json() == forecast

    with TestClient(create_service()) as client:
        assert client.get("/health").json()["model_loaded"] is True
        assert client.get(f"/train/{run_id}").json()["model_version"] == "1"
        assert client.get(f"/train/{next_run_id}").json()["model_version"] == "2"
        assert client.post("/predict", json=request).json() == forecast
