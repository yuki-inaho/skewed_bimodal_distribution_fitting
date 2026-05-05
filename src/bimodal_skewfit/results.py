"""Shared result types and helper utilities for fitted density models.

(layer: kernel) Carries the canonical :class:`FitResult` dataclass plus pure
statistics helpers (sample skewness, robust location/scale, parameter bounds).
The ``logpdf``/``pdf`` methods on :class:`FitResult` simply re-dispatch to the
kernels in :mod:`distributions` via the registry — no optimization or I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .distributions import FloatArray
from .registry import DISTRIBUTIONS


@dataclass(frozen=True, slots=True)
class FitResult:
    """Fitted model parameters and scalar diagnostics."""

    model: str
    params: dict[str, float]
    nll: float
    aic: float
    bic: float
    converged: bool
    message: str
    nobs: int
    n_parameters: int

    @property
    def loglik(self) -> float:
        """Return maximized log-likelihood."""

        return -self.nll

    def logpdf(self, x: ArrayLike) -> FloatArray:
        """Evaluate the fitted model's log-density."""

        spec = DISTRIBUTIONS[self.model]
        values = [self.params[name] for name in spec.param_names]
        return spec.logpdf(x, *values)

    def pdf(self, x: ArrayLike) -> FloatArray:
        """Evaluate the fitted model's density."""

        return np.exp(self.logpdf(x))


def finite_nll(logpdf: FloatArray) -> float:
    """Convert a log-density vector to a finite negative log-likelihood."""

    if logpdf.size == 0 or not np.all(np.isfinite(logpdf)):
        return float("inf")
    nll = -float(np.sum(logpdf))
    if not np.isfinite(nll):
        return float("inf")
    return nll


def robust_location_scale(x: FloatArray) -> tuple[float, float]:
    """Return robust location and positive scale starting values."""

    loc = float(np.median(x))
    q25, q75 = np.quantile(x, [0.25, 0.75])
    iqr_scale = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
    std_scale = float(np.std(x, ddof=1)) if x.size > 1 else 1.0
    scale = max(iqr_scale, std_scale * 0.5, 1e-3)
    return loc, scale


def sample_skewness(x: FloatArray) -> float:
    """Compute a stable sample skewness diagnostic."""

    centered = x - np.mean(x)
    sd = float(np.std(centered, ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(np.mean((centered / sd) ** 3))


def bounds_for_data(x: FloatArray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Build conservative location and log-scale bounds from empirical support."""

    sd = max(float(np.std(x, ddof=1)), 1e-3)
    data_min = float(np.min(x))
    data_max = float(np.max(x))
    span = max(data_max - data_min, sd, 1.0)
    loc_bounds = (data_min - 2.0 * span, data_max + 2.0 * span)
    log_scale_bounds = (np.log(max(1e-3 * sd, 1e-6)), np.log(max(10.0 * span, 1.0)))
    return loc_bounds, log_scale_bounds
