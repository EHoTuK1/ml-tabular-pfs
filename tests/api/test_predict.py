from typing import Any

import pytest
from fastapi.testclient import TestClient

from pfs.api.schemas import MAX_ITEMS

from .conftest import FakePredictor


def test_single_pair(client: TestClient) -> None:
    r = client.post("/predict", json={"items": [{"shop_id": 1, "item_id": 10}]})
    assert r.status_code == 200
    assert r.json() == {"predictions": [{"shop_id": 1, "item_id": 10, "item_cnt_month": 11.0}]}


def test_batch_keeps_order_and_duplicates(client: TestClient, predictor: FakePredictor) -> None:
    items = [
        {"shop_id": 2, "item_id": 20},
        {"shop_id": 1, "item_id": 10},
        {"shop_id": 2, "item_id": 20},
    ]
    r = client.post("/predict", json={"items": items})
    assert r.status_code == 200
    preds: list[dict[str, Any]] = r.json()["predictions"]
    assert [(p["shop_id"], p["item_id"]) for p in preds] == [(2, 20), (1, 10), (2, 20)]
    assert [p["item_cnt_month"] for p in preds] == [22.0, 11.0, 22.0]
    assert len(predictor.calls) == 1  # весь пакет одним вызовом


def test_unknown_ids_400(client: TestClient) -> None:
    items = [{"shop_id": 1, "item_id": 10}, {"shop_id": 99, "item_id": 777}]
    r = client.post("/predict", json={"items": items})
    assert r.status_code == 400
    assert r.json()["detail"] == {"message": "unknown ids", "shop_id": [99], "item_id": [777]}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"items": []},
        {"items": [{"shop_id": 1}]},
        {"items": [{"shop_id": "abc", "item_id": 10}]},
        {"items": [{"shop_id": -1, "item_id": 10}]},
        {"items": [{"shop_id": 1.5, "item_id": 10}]},
        {"shop_id": 1, "item_id": 10},
    ],
)
def test_invalid_body_422(
    client: TestClient, predictor: FakePredictor, body: dict[str, Any]
) -> None:
    r = client.post("/predict", json=body)
    assert r.status_code == 422
    assert predictor.calls == []


def test_too_many_items_422(client: TestClient) -> None:
    items = [{"shop_id": 1, "item_id": 10}] * (MAX_ITEMS + 1)
    r = client.post("/predict", json={"items": items})
    assert r.status_code == 422


def test_no_model_503(client_no_model: TestClient) -> None:
    r = client_no_model.post("/predict", json={"items": [{"shop_id": 1, "item_id": 10}]})
    assert r.status_code == 503


def test_get_not_allowed(client: TestClient) -> None:
    assert client.get("/predict").status_code == 405
