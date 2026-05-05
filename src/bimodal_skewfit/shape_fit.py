"""Likelihood fitters for the four skewed/bimodal single-family models.

(layer: driver) The four public ``fit_*`` entry points are uniform thin
wrappers over a single generic optimizer driver. Each model contributes a
small immutable :class:`TransformedModelSpec` that describes (a) how raw
L-BFGS-B coordinates are transformed into physical parameters, (b) the bounds
for each shape parameter, and (c) how starting points are seeded from the
data. This shape mirrors the data-driven dispatch a Rust port would use — one
``enum`` variant per model, each carrying its own struct of transforms and
seeds — and keeps the Python side free of ad-hoc duplication.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from .distributions import FloatArray, as_float_array
from .evaluate import aic, bic
from .registry import DISTRIBUTIONS
from .results import FitResult, bounds_for_data, finite_nll, robust_location_scale, sample_skewness

Bounds = list[tuple[float, float]]
LocScaleStarts = list[tuple[float, float]]
UnpackFn = Callable[[FloatArray], tuple[float, ...]]
StartsFactory = Callable[[FloatArray, LocScaleStarts], list[FloatArray]]


@dataclass(frozen=True, slots=True)
class TransformedModelSpec:
    """Per-model recipe for the transformed-parameter L-BFGS-B fitter."""

    model_name: str
    unpack: UnpackFn
    shape_bounds: tuple[tuple[float, float], ...]
    starts_factory: StartsFactory


def _fit_transformed_model(
    x: FloatArray,
    model: str,
    starts: list[FloatArray],
    unpack: UnpackFn,
    bounds: Bounds,
    max_iter: int,
) -> FitResult:
    """Fit a transformed-parameter likelihood with bounded L-BFGS-B."""

    spec = DISTRIBUTIONS[model]

    def objective(raw: FloatArray) -> float:
        params = unpack(raw)
        return finite_nll(spec.logpdf(x, *params))

    best_result = None
    for start in starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "ftol": 1e-8,
                "gtol": 1e-5,
                "maxfun": max(80, max_iter * 8),
                "maxiter": max_iter,
                "maxls": 20,
            },
        )
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result

    if best_result is None:
        raise RuntimeError(f"{model} optimization produced no result")

    final_params = unpack(np.asarray(best_result.x, dtype=np.float64))
    nll = finite_nll(spec.logpdf(x, *final_params))
    return FitResult(
        model=model,
        params={
            name: float(value) for name, value in zip(spec.param_names, final_params, strict=True)
        },
        nll=nll,
        aic=aic(nll, spec.num_parameters),
        bic=bic(nll, spec.num_parameters, x.size),
        converged=bool(best_result.success and np.isfinite(nll)),
        message=str(best_result.message),
        nobs=x.size,
        n_parameters=spec.num_parameters,
    )


def _common_location_scale_shape_starts(
    x: FloatArray,
) -> tuple[LocScaleStarts, tuple[tuple[float, float], tuple[float, float]]]:
    """Build shared location/scale starts for shape-parameter models."""

    median_loc, robust_scale = robust_location_scale(x)
    mean_loc = float(np.mean(x))
    sd = max(float(np.std(x, ddof=1)), 1e-3)
    starts = [(mean_loc, np.log(sd)), (median_loc, np.log(robust_scale))]
    return starts, bounds_for_data(x)


def _unpack_abn(raw: FloatArray) -> tuple[float, float, float, float]:
    return (
        float(raw[0]),
        float(np.exp(raw[1])),
        float(np.exp(raw[2])),
        float(np.tanh(raw[3])),
    )


def _unpack_adn(raw: FloatArray) -> tuple[float, float, float, float]:
    return (float(raw[0]), float(np.exp(raw[1])), float(np.exp(raw[2])), float(raw[3]))


def _unpack_bsn_fs(raw: FloatArray) -> tuple[float, float, float, float]:
    return (
        float(raw[0]),
        float(np.exp(raw[1])),
        float(np.exp(raw[2])),
        float(np.exp(raw[3])),
    )


def _unpack_ntpn(raw: FloatArray) -> tuple[float, float, float, float]:
    return (float(raw[0]), float(np.exp(raw[1])), float(raw[2]), float(np.exp(raw[3])))


def _starts_abn(arr: FloatArray, loc_scale_starts: LocScaleStarts) -> list[FloatArray]:
    skew_sign = float(np.sign(sample_skewness(arr)))
    starts: list[FloatArray] = []
    for loc, log_scale in loc_scale_starts:
        for lam, alpha in ((0.5, 0.0), (2.0, 0.65 * skew_sign)):
            clipped_alpha = float(np.clip(alpha, -0.95, 0.95))
            starts.append(np.array([loc, log_scale, np.log(lam), np.arctanh(clipped_alpha)]))
    return starts


def _starts_adn(arr: FloatArray, loc_scale_starts: LocScaleStarts) -> list[FloatArray]:
    skew_guess = float(np.clip(sample_skewness(arr), -4.0, 4.0))
    starts: list[FloatArray] = []
    for loc, log_scale in loc_scale_starts:
        for sep, skew in ((1e-4, 0.0), (2.0, skew_guess)):
            starts.append(np.array([loc, log_scale, np.log(sep), skew]))
    return starts


def _starts_bsn_fs(arr: FloatArray, loc_scale_starts: LocScaleStarts) -> list[FloatArray]:
    skew = sample_skewness(arr)
    gamma_direction = 1.5 if skew >= 0.0 else 1.0 / 1.5
    starts: list[FloatArray] = []
    for loc, log_scale in loc_scale_starts:
        for alpha, gamma in ((1e-5, 1.0), (5.0, gamma_direction)):
            starts.append(np.array([loc, log_scale, np.log(alpha), np.log(gamma)]))
    return starts


def _starts_ntpn(arr: FloatArray, loc_scale_starts: LocScaleStarts) -> list[FloatArray]:
    skew_value = sample_skewness(arr)
    skew_sign = float(np.sign(skew_value)) if skew_value != 0.0 else 1.0
    starts: list[FloatArray] = []
    for loc, log_scale in loc_scale_starts:
        for alpha, lam in ((0.0, 1e-4), (0.75 * skew_sign, 2.0)):
            starts.append(np.array([loc, log_scale, alpha, np.log(lam)]))
    return starts


SHAPE_MODEL_SPECS: dict[str, TransformedModelSpec] = {
    "abn": TransformedModelSpec(
        model_name="abn",
        unpack=_unpack_abn,
        shape_bounds=((np.log(1e-4), np.log(10.0)), (-4.5, 4.5)),
        starts_factory=_starts_abn,
    ),
    "adn": TransformedModelSpec(
        model_name="adn",
        unpack=_unpack_adn,
        shape_bounds=((np.log(1e-5), np.log(10.0)), (-20.0, 20.0)),
        starts_factory=_starts_adn,
    ),
    "bsn_fs": TransformedModelSpec(
        model_name="bsn_fs",
        unpack=_unpack_bsn_fs,
        shape_bounds=((np.log(1e-8), np.log(50.0)), (-4.0, 4.0)),
        starts_factory=_starts_bsn_fs,
    ),
    "ntpn": TransformedModelSpec(
        model_name="ntpn",
        unpack=_unpack_ntpn,
        shape_bounds=((-10.0, 10.0), (np.log(1e-5), np.log(10.0))),
        starts_factory=_starts_ntpn,
    ),
}


def _fit_shape_model(model_name: str, x: ArrayLike, max_iter: int) -> FitResult:
    """Generic driver for any registered transformed-parameter model."""

    spec = SHAPE_MODEL_SPECS[model_name]
    arr = as_float_array(x)
    loc_scale_starts, (loc_bounds, log_scale_bounds) = _common_location_scale_shape_starts(arr)
    starts = spec.starts_factory(arr, loc_scale_starts)
    bounds = [loc_bounds, log_scale_bounds, *spec.shape_bounds]
    return _fit_transformed_model(arr, model_name, starts, spec.unpack, bounds, max_iter)


def fit_abn(x: ArrayLike, max_iter: int = 250) -> FitResult:
    """Fit ABN by transformed-parameter L-BFGS-B."""

    return _fit_shape_model("abn", x, max_iter)


def fit_adn(x: ArrayLike, max_iter: int = 250) -> FitResult:
    """Fit ADN by transformed-parameter L-BFGS-B."""

    return _fit_shape_model("adn", x, max_iter)


def fit_bsn_fs(x: ArrayLike, max_iter: int = 250) -> FitResult:
    """Fit the Fernandez-Steel bimodal skew-normal distribution."""

    return _fit_shape_model("bsn_fs", x, max_iter)


def fit_ntpn(x: ArrayLike, max_iter: int = 250) -> FitResult:
    """Fit NTPN by transformed-parameter L-BFGS-B."""

    return _fit_shape_model("ntpn", x, max_iter)
