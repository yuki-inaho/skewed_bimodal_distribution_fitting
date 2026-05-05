"""Model evaluation utilities for fitted one-dimensional densities.

(layer: kernel) Pure helpers — AIC/BIC, ISE, mode counting on a density grid.
No I/O, no RNG. These are the second tier (after :mod:`distributions`) that a
Rust port would lift verbatim as free functions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import find_peaks

from .distributions import FloatArray, as_float_array


def aic(nll: float, k: int) -> float:
    """Compute Akaike's information criterion from negative log-likelihood."""

    return 2.0 * k + 2.0 * nll


def bic(nll: float, k: int, n: int) -> float:
    """Compute Bayesian information criterion from negative log-likelihood."""

    return float(k) * np.log(float(n)) + 2.0 * nll


def count_density_modes(x_grid: ArrayLike, density: ArrayLike, min_prominence: float = 0.02) -> int:
    """Count visually meaningful local maxima on an evaluation grid.

    This is a **grid-based visual diagnostic**, not a strict modality decision
    rule. The prominence threshold is relative to the maximum density and
    suppresses numerical wiggles that do not matter for the unimodal/bimodal
    distinction. Weakly separated bimodality, shoulder densities, and ridges
    near the modality boundary may be classified as a single mode by this
    routine even when a tighter analytic test would call them bimodal.
    """

    xs = as_float_array(x_grid)
    ys = as_float_array(density)
    if xs.size != ys.size or xs.size < 5:
        raise ValueError("x_grid and density must have the same length >= 5")
    if not np.all(np.isfinite(ys)) or np.max(ys) <= 0.0:
        return 0

    prominence = float(np.max(ys) * min_prominence)
    peaks, _ = find_peaks(ys, prominence=prominence)
    return len(peaks)


def empirical_quantile_grid(x: ArrayLike, padding: float = 0.25, num: int = 600) -> FloatArray:
    """Create an evaluation grid around empirical support."""

    arr = as_float_array(x)
    low = float(np.min(arr))
    high = float(np.max(arr))
    span = max(high - low, 1e-6)
    return np.linspace(low - padding * span, high + padding * span, num, dtype=np.float64)


def integrated_squared_error(
    x_grid: ArrayLike,
    estimated_density: ArrayLike,
    reference_density: ArrayLike,
) -> float:
    """Approximate integrated squared error on a one-dimensional grid."""

    xs = as_float_array(x_grid)
    estimated = as_float_array(estimated_density)
    reference = as_float_array(reference_density)
    if xs.size != estimated.size or xs.size != reference.size:
        raise ValueError("x_grid, estimated_density, and reference_density sizes must match")
    return float(np.trapezoid((estimated - reference) ** 2, xs))
