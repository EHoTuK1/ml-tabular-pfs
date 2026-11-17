from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from pfs.api import create_app
from pfs.api.predictor import UnknownIdError


class FakePredictor:
    """Заглушка вместо Processor+Model: известны магазины {1,2}, товары {10,20}."""

    shops = {1, 2}
    items = {10, 20}

    def __init__(self) -> None:
        self.calls: list[list[tuple[int, int]]] = []

    def predict(self, pairs: Sequence[tuple[int, int]]) -> list[float]:
        self.calls.append(list(pairs))
        bad_s = [s for s, _ in pairs if s not in self.shops]
        bad_i = [i for _, i in pairs if i not in self.items]
        if bad_s or bad_i:
            raise UnknownIdError(bad_s, bad_i)
        return [float(s + i) for s, i in pairs]


@pytest.fixture
def predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def client(predictor: FakePredictor) -> TestClient:
    return TestClient(create_app(predictor))


@pytest.fixture
def client_no_model() -> TestClient:
    return TestClient(create_app(None))
