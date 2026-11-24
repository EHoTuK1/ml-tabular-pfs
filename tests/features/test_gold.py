"""Сверяет сборку Gold с офлайн-подготовкой исходных CSV."""

from pathlib import Path

import pandas as pd

from pfs.features.gold import build_gold

from .conftest import Raw, load_tmp_script


def test_gold_matches_legacy(raw: Raw, tmp_path: Path) -> None:
    """Сохраняет дедупликацию, переобозначение магазинов, возвраты и clip."""
    path = tmp_path / "sales_train.csv"
    raw.sales.to_csv(path, index=False, date_format="%d.%m.%Y")
    sales = pd.read_csv(path, parse_dates=["date"], date_format="%d.%m.%Y")

    expected = load_tmp_script("prepare_data").prepare_raw_data(path)
    pd.testing.assert_frame_equal(build_gold(sales), expected)
