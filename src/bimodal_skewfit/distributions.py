"""Probability-density kernels for skewed and bimodal fitting experiments.

(layer: kernel) The numerical core is intentionally composed of stateless,
side-effect-free functions. That shape is useful for future migration to Rust
or C++ because each function has explicit scalar parameters, array input, and
array output. The ``name -> spec`` registry that pairs each kernel with its
public name and parameter signature lives in :mod:`bimodal_skewfit.registry`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import log_ndtr, logsumexp

LOG_TWO = float(np.log(2.0))
LOG_SQRT_2PI = float(0.5 * np.log(2.0 * np.pi))
FloatArray = NDArray[np.float64]
LogPdf = Callable[..., FloatArray]


def as_float_array(x: ArrayLike) -> FloatArray:
    """Convert scalar or array-like input to a one-dimensional float64 array."""

    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.ravel()


def log_phi(x: ArrayLike) -> FloatArray:
    """Evaluate the standard normal log-density."""

    z = as_float_array(x)
    return -0.5 * z * z - LOG_SQRT_2PI


def location_scale_z(x: ArrayLike, loc: float, scale: float) -> FloatArray:
    """Standardize observations and reject invalid scale values explicitly."""

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be a positive finite number")
    return (as_float_array(x) - loc) / scale


def normal_logpdf(x: ArrayLike, loc: float, scale: float) -> FloatArray:
    """Evaluate a location-scale normal log-density."""

    z = location_scale_z(x, loc, scale)
    return log_phi(z) - np.log(scale)


def gmm2_logpdf(
    x: ArrayLike,
    weight: float,
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
) -> FloatArray:
    """Evaluate an unconstrained two-component Gaussian mixture log-density."""

    x_arr = as_float_array(x)
    if not (0.0 < weight < 1.0):
        return np.full_like(x_arr, -np.inf, dtype=np.float64)
    if sigma1 <= 0.0 or sigma2 <= 0.0:
        return np.full_like(x_arr, -np.inf, dtype=np.float64)

    comp1 = np.log(weight) + normal_logpdf(x_arr, mu1, sigma1)
    comp2 = np.log1p(-weight) + normal_logpdf(x_arr, mu2, sigma2)
    return logsumexp(np.vstack([comp1, comp2]), axis=0)


def abn_logpdf(x: ArrayLike, loc: float, scale: float, lam: float, alpha: float) -> FloatArray:
    """Evaluate the Asymmetric Bimodal Normal log-density.

    Standard form:
        Z ~ w N(+lambda, 1) + (1 - w) N(-lambda, 1),
        w = (1 + alpha) / 2, lambda > 0, -1 < alpha < 1.
    """

    z = location_scale_z(x, loc, scale)
    if lam <= 0.0 or not (-1.0 < alpha < 1.0):
        return np.full_like(z, -np.inf, dtype=np.float64)

    w_pos = 0.5 * (1.0 + alpha)
    w_neg = 0.5 * (1.0 - alpha)
    comp_pos = np.log(w_pos) + log_phi(z - lam)
    comp_neg = np.log(w_neg) + log_phi(z + lam)
    return logsumexp(np.vstack([comp_pos, comp_neg]), axis=0) - np.log(scale)


def adn_logpdf(x: ArrayLike, loc: float, scale: float, sep: float, skew: float) -> FloatArray:
    """Evaluate the Asymmetric Double Normal log-density.

    Implemented standard density:
        f(z; a, lambda) = [phi(z - a) + phi(z + a)] * Phi(lambda z).

    It reduces to the skew-normal density when ``a = 0`` and to an equal-weight
    two-normal mixture when ``lambda = 0``.
    """

    z = location_scale_z(x, loc, scale)
    if sep < 0.0 or not np.isfinite(skew):
        return np.full_like(z, -np.inf, dtype=np.float64)

    symmetric_kernel = logsumexp(np.vstack([log_phi(z - sep), log_phi(z + sep)]), axis=0)
    return symmetric_kernel + log_ndtr(skew * z) - np.log(scale)


def bsn_fs_b_gamma(gamma: float) -> float:
    """Return the second moment of the Fernandez-Steel skewed normal kernel."""

    return float((gamma**3 + gamma**-3) / (gamma + gamma**-1))


def bsn_fs_logpdf(
    x: ArrayLike,
    loc: float,
    scale: float,
    alpha: float,
    gamma: float,
) -> FloatArray:
    """Evaluate the Fernandez-Steel bimodal skew-normal log-density."""

    z = location_scale_z(x, loc, scale)
    if alpha < 0.0 or gamma <= 0.0:
        return np.full_like(z, -np.inf, dtype=np.float64)

    b_gamma = bsn_fs_b_gamma(gamma)
    log_skew_kernel = np.empty_like(z, dtype=np.float64)
    positive = z >= 0.0
    log_const = np.log(2.0) - np.log(gamma + gamma**-1) - LOG_SQRT_2PI
    log_skew_kernel[positive] = log_const - 0.5 * (z[positive] / gamma) ** 2
    log_skew_kernel[~positive] = log_const - 0.5 * (z[~positive] * gamma) ** 2
    log_tilt = np.log1p(alpha * z * z) - np.log1p(alpha * b_gamma)
    return log_tilt + log_skew_kernel - np.log(scale)


def ntpn_logpdf(x: ArrayLike, loc: float, scale: float, alpha: float, lam: float) -> FloatArray:
    """Evaluate the New Two-Piece Normal extension log-density.

    The implementation uses the paper's polynomial tilt of the two-piece-normal
    kernel and its closed-form normalizing constant.
    """

    z = location_scale_z(x, loc, scale)
    if lam < 0.0 or not np.isfinite(alpha):
        return np.full_like(z, -np.inf, dtype=np.float64)

    tpn_kernel = logsumexp(np.vstack([log_phi(z - lam), log_phi(z + lam)]), axis=0) - LOG_TWO
    c_norm = alpha * alpha * (lam * lam + 1.0) + 2.0
    log_tilt = np.log((1.0 - alpha * z) ** 2 + 1.0) - np.log(c_norm)
    return log_tilt + tpn_kernel - np.log(scale)


def pdf_from_logpdf(logpdf_fn: LogPdf, x: ArrayLike, *params: float) -> FloatArray:
    """Evaluate a density by exponentiating a supplied log-density."""

    return np.exp(logpdf_fn(x, *params))
