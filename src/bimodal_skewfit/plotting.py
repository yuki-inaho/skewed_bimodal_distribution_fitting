"""Plotting helpers for experiment outputs.

(layer: io) The matplotlib boundary. The active backend is left untouched.
This package is intended for use from interactive shells and notebooks, so
forcibly switching the backend on import would override a user-selected GUI
backend. ``Figure.savefig`` works regardless of the active backend, which is
all that the orchestration layer needs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from .distributions import FloatArray
from .evaluate import empirical_quantile_grid
from .fit import FitResult


def plot_fit_overlay(
    x: ArrayLike,
    results: list[FitResult],
    output_path: Path,
    title: str,
    max_models: int = 4,
    reference_grid: FloatArray | None = None,
    reference_density: FloatArray | None = None,
) -> None:
    """Write a histogram and fitted density overlay plot."""

    arr = np.asarray(x, dtype=np.float64)
    grid = empirical_quantile_grid(arr) if reference_grid is None else reference_grid
    ranked = sorted(results, key=lambda result: result.bic)[:max_models]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(arr, bins="fd", density=True, alpha=0.35, label="empirical histogram")
    if reference_density is not None:
        ax.plot(grid, reference_density, linewidth=2, linestyle="--", label="true density")
    for result in ranked:
        ax.plot(grid, result.pdf(grid), linewidth=2, label=f"{result.model} (BIC={result.bic:.1f})")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
