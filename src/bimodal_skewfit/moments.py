"""Closed-form moment summaries for supported skewed bimodal families.

(layer: kernel) These functions mirror the density parameterization in
:mod:`bimodal_skewfit.distributions`. Raw moments are evaluated in standard
form first, then location-scale transforms are applied analytically.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, isfinite, pi, sqrt
from math import gamma as gamma_fn


@dataclass(frozen=True, slots=True)
class MomentSummary:
    """First four moments and shape coefficients for one distribution.

    ``raw_moments`` stores orders 1 through 4. ``central_moments`` stores
    central orders 2 through 4.
    """

    mean: float
    variance: float
    skewness: float
    kurtosis: float
    excess_kurtosis: float
    bimodal_coefficient: float
    raw_moments: tuple[float, float, float, float]
    central_moments: tuple[float, float, float]


def ntpn_tpn_raw_moment(order: int, lam: float) -> float:
    """Return ``E[Z**order]`` for the symmetric TPN base of NTPN."""

    _validate_order(order)
    if not isfinite(lam) or lam < 0.0:
        raise ValueError("lam must be a non-negative finite number")
    if order % 2 == 1:
        return 0.0

    moment = 0.0
    for normal_order in range(0, order + 1, 2):
        lam_order = order - normal_order
        moment += (
            comb(order, normal_order) * _standard_normal_raw_moment(normal_order) * lam**lam_order
        )
    return float(moment)


def ntpn_standard_raw_moment(order: int, alpha: float, lam: float) -> float:
    """Return the standard-form NTPN raw moment of integer ``order``."""

    _validate_order(order)
    _validate_ntpn_params(alpha, lam)
    c_norm = 2.0 + alpha * alpha * ntpn_tpn_raw_moment(2, lam)
    numerator = (
        2.0 * ntpn_tpn_raw_moment(order, lam)
        - 2.0 * alpha * ntpn_tpn_raw_moment(order + 1, lam)
        + alpha * alpha * ntpn_tpn_raw_moment(order + 2, lam)
    )
    return float(numerator / c_norm)


def ntpn_moments(
    alpha: float,
    lam: float,
    *,
    loc: float = 0.0,
    scale: float = 1.0,
) -> MomentSummary:
    """Return mean, variance, skewness, kurtosis, and BC for NTPN."""

    _validate_ntpn_params(alpha, lam)
    standard_raw = tuple(ntpn_standard_raw_moment(order, alpha, lam) for order in range(5))
    return _summarize_location_scale_moments(standard_raw, loc, scale)


def bsn_fs_skew_raw_moment(order: int, gamma: float) -> float:
    """Return ``E[Z**order]`` for the Fernandez-Steel skew-normal base."""

    _validate_order(order)
    if not isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be a positive finite number")

    sign = -1.0 if order % 2 == 1 else 1.0
    abs_normal_moment = 2.0 ** (0.5 * order) * gamma_fn(0.5 * (order + 1)) / sqrt(pi)
    numerator = gamma ** (order + 1) + sign * gamma ** (-(order + 1))
    denominator = gamma + gamma**-1
    return float(abs_normal_moment * numerator / denominator)


def bsn_fs_standard_raw_moment(order: int, alpha: float, gamma: float) -> float:
    """Return the standard-form BSN-FS raw moment of integer ``order``."""

    _validate_order(order)
    _validate_bsn_fs_params(alpha, gamma)
    h2 = bsn_fs_skew_raw_moment(2, gamma)
    denominator = 1.0 + alpha * h2
    numerator = bsn_fs_skew_raw_moment(order, gamma) + alpha * bsn_fs_skew_raw_moment(
        order + 2,
        gamma,
    )
    return float(numerator / denominator)


def bsn_fs_moments(
    alpha: float,
    gamma: float,
    *,
    loc: float = 0.0,
    scale: float = 1.0,
) -> MomentSummary:
    """Return mean, variance, skewness, kurtosis, and BC for BSN-FS."""

    _validate_bsn_fs_params(alpha, gamma)
    standard_raw = tuple(bsn_fs_standard_raw_moment(order, alpha, gamma) for order in range(5))
    return _summarize_location_scale_moments(standard_raw, loc, scale)


def _summarize_location_scale_moments(
    standard_raw: tuple[float, ...],
    loc: float,
    scale: float,
) -> MomentSummary:
    _validate_location_scale(loc, scale)
    if len(standard_raw) < 5:
        raise ValueError("standard_raw must contain moments from order 0 through 4")

    standard_mean = standard_raw[1]
    standard_mu2, standard_mu3, standard_mu4 = _central_moments_2_to_4(
        standard_raw[1],
        standard_raw[2],
        standard_raw[3],
        standard_raw[4],
    )
    mu2 = scale * scale * standard_mu2
    mu3 = scale**3 * standard_mu3
    mu4 = scale**4 * standard_mu4
    raw_moments = tuple(
        _location_scale_raw_moment(order, standard_raw, loc, scale) for order in range(1, 5)
    )
    skewness = mu3 / (mu2**1.5)
    kurtosis = mu4 / (mu2 * mu2)
    bimodal_coefficient = (mu2**3 + mu3 * mu3) / (mu2 * mu4)
    return MomentSummary(
        mean=float(loc + scale * standard_mean),
        variance=float(mu2),
        skewness=float(skewness),
        kurtosis=float(kurtosis),
        excess_kurtosis=float(kurtosis - 3.0),
        bimodal_coefficient=float(bimodal_coefficient),
        raw_moments=(
            float(raw_moments[0]),
            float(raw_moments[1]),
            float(raw_moments[2]),
            float(raw_moments[3]),
        ),
        central_moments=(float(mu2), float(mu3), float(mu4)),
    )


def _central_moments_2_to_4(
    m1: float, m2: float, m3: float, m4: float
) -> tuple[float, float, float]:
    mu2 = m2 - m1 * m1
    mu3 = m3 - 3.0 * m1 * m2 + 2.0 * m1**3
    mu4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1**4
    if mu2 <= 0.0 or mu4 <= 0.0:
        raise ValueError("moments imply a non-positive variance or fourth central moment")
    return float(mu2), float(mu3), float(mu4)


def _location_scale_raw_moment(
    order: int,
    standard_raw: tuple[float, ...],
    loc: float,
    scale: float,
) -> float:
    return float(
        sum(
            comb(order, standard_order)
            * loc ** (order - standard_order)
            * scale**standard_order
            * standard_raw[standard_order]
            for standard_order in range(order + 1)
        ),
    )


def _standard_normal_raw_moment(order: int) -> float:
    if order % 2 == 1:
        return 0.0
    moment = 1.0
    for value in range(1, order, 2):
        moment *= value
    return float(moment)


def _validate_order(order: int) -> None:
    if not isinstance(order, int) or order < 0:
        raise ValueError("order must be a non-negative integer")


def _validate_location_scale(loc: float, scale: float) -> None:
    if not isfinite(loc):
        raise ValueError("loc must be finite")
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be a positive finite number")


def _validate_ntpn_params(alpha: float, lam: float) -> None:
    if not isfinite(alpha):
        raise ValueError("alpha must be finite")
    if not isfinite(lam) or lam < 0.0:
        raise ValueError("lam must be a non-negative finite number")


def _validate_bsn_fs_params(alpha: float, gamma: float) -> None:
    if not isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a non-negative finite number")
    if not isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be a positive finite number")


__all__ = [
    "MomentSummary",
    "bsn_fs_moments",
    "bsn_fs_skew_raw_moment",
    "bsn_fs_standard_raw_moment",
    "ntpn_moments",
    "ntpn_standard_raw_moment",
    "ntpn_tpn_raw_moment",
]
