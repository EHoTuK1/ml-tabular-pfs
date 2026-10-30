"""Метрики и кривые обучения для прогноза месячных продаж (таргет обрезан до [0, 20])."""

from collections.abc import Callable, Mapping
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

CLIP = (0.0, 20.0)
Metric = Callable[[ArrayLike, ArrayLike], float]


def _arr(x: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64)


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Основная метрика соревнования."""
    return float(np.sqrt(np.mean((_arr(y_true) - _arr(y_pred)) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Средняя абсолютная ошибка: менее чувствительна к редким большим промахам, чем RMSE."""
    return float(np.mean(np.abs(_arr(y_true) - _arr(y_pred))))


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Средняя ошибка прогноза: > 0 модель завышает, < 0 занижает."""
    return float(np.mean(_arr(y_pred) - _arr(y_true)))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Доля объяснённой дисперсии относительно прогноза средним."""
    y, p = _arr(y_true), _arr(y_pred)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def rmse_nonzero(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """RMSE только на парах с продажами (~15% строк), где ошибается модель на самом деле."""
    y, p = _arr(y_true), _arr(y_pred)
    mask = y > 0
    return rmse(y[mask], p[mask]) if mask.any() else float("nan")


METRICS: dict[str, Metric] = {
    "rmse": rmse,
    "mae": mae,
    "bias": bias,
    "r2": r2,
    "rmse_nonzero": rmse_nonzero,
}


def score(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Все метрики для одного прогноза; прогноз обрезается до [0, 20], как таргет."""
    p = np.clip(_arr(y_pred), *CLIP)
    return {name: f(y_true, p) for name, f in METRICS.items()}


def learning_history(model: Any, X: pd.DataFrame, y: ArrayLike, metric: Metric = rmse) -> list[float]:
    """Метрика после каждой итерации бустинга (нужен staged_predict)."""
    return [metric(y, np.clip(p, *CLIP)) for p in model.staged_predict(X)]


def recorded_history(model: Any, metric: str = "rmse") -> dict[str, list[float]]:
    """Кривые, которые LightGBM записал при fit(eval_set=..., eval_names=..., eval_metric="rmse").

    Прогноз в них не обрезан до [0, 20], поэтому RMSE на них чуть выше, чем в таблице.
    """
    evals: dict[str, dict[str, list[float]]] = getattr(model, "evals_result_", None) or {}
    return {name: list(res[metric]) for name, res in evals.items() if metric in res}


Backend = Literal["plt", "plotly"]
Y_LABEL = "RMSE (прогноз обрезан до [0, 20])"


def plot_history(
    history: Mapping[str, list[float]],
    baselines: Mapping[str, float] | None = None,
    best_on: str | None = "valid",
    model_name: str | None = None,
    backend: Backend = "plt",
) -> Any:
    """Кривые метрики по итерациям, бейзлайны горизонтальными линиями, лучшая итерация по best_on.

    backend="plt" возвращает matplotlib Figure (для отчёта),
    backend="plotly" возвращает plotly Figure (подсказки с координатами при наведении).
    """
    baselines = baselines or {}
    title = f"{model_name}: кривые обучения" if model_name else "Кривые обучения"
    best: tuple[int, float] | None = None
    if best_on is not None and best_on in history:
        i = int(np.argmin(history[best_on]))
        best = (i + 1, history[best_on][i])
    if backend == "plotly":
        return _plot_plotly(history, baselines, best, best_on, title)
    return _plot_plt(history, baselines, best, best_on, title)


def _plot_plt(
    history: Mapping[str, list[float]],
    baselines: Mapping[str, float],
    best: tuple[int, float] | None,
    best_on: str | None,
    title: str,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, curve in history.items():
        ax.plot(np.arange(1, len(curve) + 1), curve, label=name)
    # цвета бейзлайнов идут после цветов кривых, чтобы не совпадать с ними
    for i, (name, value) in enumerate(baselines.items(), start=len(history)):
        ax.axhline(value, ls="--", lw=1, c=f"C{i % 10}", label=f"{name} = {value:.3f}")
    if best is not None:
        ax.axvline(best[0], ls=":", c="red", label=f"best_iter ({best_on}) = {best[0]}, {best[1]:.3f}")
    ax.set(xlabel="итерация", ylabel=Y_LABEL, title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def _plot_plotly(
    history: Mapping[str, list[float]],
    baselines: Mapping[str, float],
    best: tuple[int, float] | None,
    best_on: str | None,
    title: str,
) -> Any:
    import plotly.graph_objects as go  # импорт здесь: plotly нужен только для этого режима
    from plotly.colors import qualitative

    palette = qualitative.Plotly
    n_iter = max((len(c) for c in history.values()), default=1)
    x = np.arange(1, n_iter + 1)
    fig = go.Figure()
    for i, (name, curve) in enumerate(history.items()):
        fig.add_scatter(x=x[: len(curve)], y=curve, name=name, mode="lines",
                        line={"color": palette[i % len(palette)]})
    # бейзлайны линиями-трассами, а не add_hline: так они есть в легенде и в подсказке
    for i, (name, value) in enumerate(baselines.items(), start=len(history)):
        fig.add_scatter(x=x, y=np.full(n_iter, value), name=f"{name} = {value:.3f}", mode="lines",
                        line={"color": palette[i % len(palette)], "dash": "dash", "width": 1})
    if best is not None:
        fig.add_vline(x=best[0], line_dash="dot", line_color="red",
                      annotation_text=f"best_iter ({best_on}) = {best[0]}, {best[1]:.3f}")
    fig.update_layout(title=title, xaxis_title="итерация", yaxis_title=Y_LABEL,
                      hovermode="x unified", template="plotly_white", width=1000, height=500)
    return fig


def calculate_metrics(
    model: Any,
    splits: Mapping[str, tuple[pd.DataFrame, pd.Series]],
    model_name: str | None = None,
    baselines: Mapping[str, ArrayLike] | None = None,
    baseline_split: str = "valid",
    sample: int | None = 500_000,
    plot: bool = True,
    backend: Backend = "plt",
    seed: int = 42,
) -> pd.DataFrame:
    """Метрики модели на каждом разбиении и бейзлайнов на baseline_split, плюс график обучения.

    splits: {"train": (X, y), "valid": (X, y)}. Разбиения больше sample строк
    прорежаются случайной подвыборкой (форма кривых от этого не меняется).
    Кривые: staged_predict у sklearn-бустинга, evals_result_ у LightGBM (имена eval_names
    должны совпадать с ключами splits); у линейных моделей кривых нет, рисуется только таблица.
    baselines: {"naive": прогноз}, прогнозы выровнены со строками baseline_split.
    backend: "plt" (по умолчанию) или "plotly".
    """
    rows: dict[str, dict[str, float]] = {}
    recorded = recorded_history(model)
    history: dict[str, list[float]] = {}
    for name, (X, y) in splits.items():
        if sample is not None and len(X) > sample:
            idx = np.random.default_rng(seed).choice(len(X), sample, replace=False)
            X, y = X.iloc[idx], y.iloc[idx]
        if hasattr(model, "staged_predict"):
            history[name] = learning_history(model, X, y)
        elif name in recorded:
            history[name] = recorded[name]
        rows[name] = score(y, model.predict(X))

    base_rmse: dict[str, float] = {}
    if baselines:
        y_ref = splits[baseline_split][1]
        for name, pred in baselines.items():
            key = f"{baseline_split}: {name}"
            rows[key] = score(y_ref, pred)
            base_rmse[key] = rows[key]["rmse"]

    if plot and history:
        best_on = baseline_split if baseline_split in history else None
        fig = plot_history(history, base_rmse, best_on=best_on, model_name=model_name, backend=backend)
        if backend == "plotly":
            fig.show()
        else:
            plt.show()
    return pd.DataFrame(rows).T
