"""Closed-form normal fitting and EM fitting for K=2 Gaussian mixtures.

(layer: driver) Sits between the kernel and the orchestration layer. Calls
into :mod:`distributions`/:mod:`registry`/:mod:`results`/:mod:`evaluate`,
plus stochastic multi-start EM with a project-supplied RNG seed. No file or
network I/O.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from .distributions import FloatArray, as_float_array, gmm2_logpdf, normal_logpdf
from .evaluate import aic, bic
from .registry import DISTRIBUTIONS
from .results import FitResult, finite_nll

Gmm2Params = tuple[float, float, float, float, float]


def fit_normal(x: ArrayLike) -> FitResult:
    """Fit the normal baseline by its closed-form maximum-likelihood estimator."""

    arr = as_float_array(x)
    loc = float(np.mean(arr))
    scale = float(np.sqrt(np.mean((arr - loc) ** 2)))
    scale = max(scale, 1e-9)
    nll = finite_nll(normal_logpdf(arr, loc, scale))
    k = DISTRIBUTIONS["normal"].num_parameters
    return FitResult(
        model="normal",
        params={"loc": loc, "scale": scale},
        nll=nll,
        aic=aic(nll, k),
        bic=bic(nll, k, arr.size),
        converged=True,
        message="closed-form MLE",
        nobs=arr.size,
        n_parameters=k,
    )


def _initial_gmm_splits(
    x: FloatArray,
    rng: np.random.Generator,
    n_random: int,
) -> list[Gmm2Params]:
    """Generate deterministic quantile starts and random center starts for EM."""

    sorted_x = np.sort(x)
    nobs = x.size
    starts: list[Gmm2Params] = []
    for fraction in (0.35, 0.5, 0.65):
        cut = int(np.clip(round(nobs * fraction), 1, nobs - 1))
        left = sorted_x[:cut]
        right = sorted_x[cut:]
        starts.append(
            (
                float(left.size / nobs),
                float(np.mean(left)),
                max(float(np.std(left)), 1e-2),
                float(np.mean(right)),
                max(float(np.std(right)), 1e-2),
            )
        )

    sd = max(float(np.std(x, ddof=1)), 1e-2)
    for _ in range(n_random):
        replace = nobs < 2
        centers = rng.choice(x, size=2, replace=replace)
        starts.append((0.5, float(np.min(centers)), sd, float(np.max(centers)), sd))
    return starts


def _em_gmm2_once(
    x: FloatArray,
    start: Gmm2Params,
    max_iter: int,
    tol: float,
    variance_floor: float,
) -> tuple[Gmm2Params, float, bool]:
    """Run one EM optimization from a single GMM2 starting point."""

    weight, mu1, sigma1, mu2, sigma2 = start
    prev_ll = -np.inf
    converged = False
    for _ in range(max_iter):
        clipped_weight = float(np.clip(weight, 1e-9, 1.0 - 1e-9))
        logp1 = np.log(clipped_weight) + normal_logpdf(x, mu1, sigma1)
        logp2 = np.log(1.0 - clipped_weight) + normal_logpdf(x, mu2, sigma2)
        logden = logsumexp(np.vstack([logp1, logp2]), axis=0)
        ll = float(np.sum(logden))
        resp1 = np.exp(logp1 - logden)
        n1 = float(np.sum(resp1))
        n2 = float(x.size - n1)
        if n1 <= 1e-6 or n2 <= 1e-6:
            break

        weight = float(np.clip(n1 / x.size, 1e-6, 1.0 - 1e-6))
        mu1 = float(np.sum(resp1 * x) / n1)
        mu2 = float(np.sum((1.0 - resp1) * x) / n2)
        var1 = float(np.sum(resp1 * (x - mu1) ** 2) / n1)
        var2 = float(np.sum((1.0 - resp1) * (x - mu2) ** 2) / n2)
        sigma1 = float(np.sqrt(max(var1, variance_floor)))
        sigma2 = float(np.sqrt(max(var2, variance_floor)))
        if np.isfinite(prev_ll) and abs(ll - prev_ll) <= tol * (1.0 + abs(prev_ll)):
            converged = True
            break
        prev_ll = ll

    if mu1 > mu2:
        weight, mu1, sigma1, mu2, sigma2 = 1.0 - weight, mu2, sigma2, mu1, sigma1
    nll = finite_nll(gmm2_logpdf(x, weight, mu1, sigma1, mu2, sigma2))
    params = (float(weight), float(mu1), float(sigma1), float(mu2), float(sigma2))
    return params, nll, converged


def fit_gmm2(
    x: ArrayLike,
    seed: int = 1234,
    n_random_starts: int = 8,
    max_iter: int = 250,
    tol: float = 1e-8,
) -> FitResult:
    """Fit an unconstrained two-component Gaussian mixture by multi-start EM."""

    arr = as_float_array(x)
    rng = np.random.default_rng(seed)
    sd = max(float(np.std(arr, ddof=1)), 1e-3)
    variance_floor = max((1e-4 * sd) ** 2, 1e-12)
    starts = _initial_gmm_splits(arr, rng, n_random_starts)
    best_params: Gmm2Params | None = None
    best_nll = float("inf")
    best_converged = False

    for start in starts:
        params, nll, converged = _em_gmm2_once(arr, start, max_iter, tol, variance_floor)
        if nll < best_nll:
            best_params = params
            best_nll = nll
            best_converged = converged

    if best_params is None:
        raise RuntimeError("GMM2 EM failed to produce any candidate")

    k = DISTRIBUTIONS["gmm2"].num_parameters
    names = DISTRIBUTIONS["gmm2"].param_names
    return FitResult(
        model="gmm2",
        params=dict(zip(names, best_params, strict=True)),
        nll=best_nll,
        aic=aic(best_nll, k),
        bic=bic(best_nll, k, arr.size),
        converged=best_converged,
        message="multi-start EM",
        nobs=arr.size,
        n_parameters=k,
    )
