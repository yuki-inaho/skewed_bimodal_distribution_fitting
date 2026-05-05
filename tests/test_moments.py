"""Closed-form moment API checks for NTPN and BSN-FS."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from bimodal_skewfit import MomentSummary, bsn_fs_moments, ntpn_moments
from bimodal_skewfit.distributions import bsn_fs_b_gamma, bsn_fs_logpdf, ntpn_logpdf
from bimodal_skewfit.moments import (
    bsn_fs_skew_raw_moment,
    ntpn_tpn_raw_moment,
)


def test_public_moment_api_returns_summary() -> None:
    summary = ntpn_moments(alpha=0.5, lam=0.5)
    assert isinstance(summary, MomentSummary)
    assert summary.mean == pytest.approx(-0.5405405405405406)
    assert summary.variance == pytest.approx(1.28214, rel=1e-5)
    assert summary.bimodal_coefficient > 0.0


def test_ntpn_tpn_base_raw_moments_match_mixture_formula() -> None:
    lam = 1.4
    assert ntpn_tpn_raw_moment(0, lam) == pytest.approx(1.0)
    assert ntpn_tpn_raw_moment(1, lam) == pytest.approx(0.0)
    assert ntpn_tpn_raw_moment(2, lam) == pytest.approx(1.0 + lam**2)
    assert ntpn_tpn_raw_moment(4, lam) == pytest.approx(3.0 + 6.0 * lam**2 + lam**4)


def test_bsn_fs_base_second_moment_matches_distribution_constant() -> None:
    for gamma in (0.6, 1.0, 1.7):
        assert bsn_fs_skew_raw_moment(2, gamma) == pytest.approx(
            bsn_fs_b_gamma(gamma),
            rel=1e-15,
        )


def test_ntpn_moments_match_numerical_integration_with_location_scale() -> None:
    loc, scale, alpha, lam = -0.3, 1.4, 0.7, 0.9
    summary = ntpn_moments(alpha=alpha, lam=lam, loc=loc, scale=scale)
    raw_quad = tuple(
        _raw_moment_from_quad(ntpn_logpdf, (loc, scale, alpha, lam), order) for order in range(1, 5)
    )
    np.testing.assert_allclose(summary.raw_moments, raw_quad, rtol=2e-8, atol=2e-8)

    central_quad = tuple(
        _central_moment_from_quad(ntpn_logpdf, (loc, scale, alpha, lam), summary.mean, order)
        for order in range(2, 5)
    )
    np.testing.assert_allclose(summary.central_moments, central_quad, rtol=2e-8, atol=2e-8)


def test_bsn_fs_moments_match_numerical_integration_with_location_scale() -> None:
    loc, scale, alpha, gamma = 0.2, 1.3, 0.6, 1.5
    summary = bsn_fs_moments(alpha=alpha, gamma=gamma, loc=loc, scale=scale)
    raw_quad = tuple(
        _raw_moment_from_quad(bsn_fs_logpdf, (loc, scale, alpha, gamma), order)
        for order in range(1, 5)
    )
    np.testing.assert_allclose(summary.raw_moments, raw_quad, rtol=2e-8, atol=2e-8)

    central_quad = tuple(
        _central_moment_from_quad(bsn_fs_logpdf, (loc, scale, alpha, gamma), summary.mean, order)
        for order in range(2, 5)
    )
    np.testing.assert_allclose(summary.central_moments, central_quad, rtol=2e-8, atol=2e-8)


def test_bsn_fs_symmetric_moment_and_bc_formula() -> None:
    alpha = 0.8
    summary = bsn_fs_moments(alpha=alpha, gamma=1.0)
    variance = (1.0 + 3.0 * alpha) / (1.0 + alpha)
    kurtosis = 3.0 * (1.0 + 5.0 * alpha) * (1.0 + alpha) / (1.0 + 3.0 * alpha) ** 2
    assert summary.mean == pytest.approx(0.0, abs=1e-15)
    assert summary.variance == pytest.approx(variance)
    assert summary.skewness == pytest.approx(0.0, abs=1e-15)
    assert summary.kurtosis == pytest.approx(kurtosis)
    assert summary.bimodal_coefficient == pytest.approx(1.0 / kurtosis)


def test_moment_shape_coefficients_are_location_scale_invariant() -> None:
    standard = ntpn_moments(alpha=-0.3, lam=1.2)
    shifted = ntpn_moments(alpha=-0.3, lam=1.2, loc=10.0, scale=2.5)
    assert shifted.mean == pytest.approx(10.0 + 2.5 * standard.mean)
    assert shifted.variance == pytest.approx(2.5**2 * standard.variance)
    assert shifted.skewness == pytest.approx(standard.skewness)
    assert shifted.kurtosis == pytest.approx(standard.kurtosis)
    assert shifted.bimodal_coefficient == pytest.approx(standard.bimodal_coefficient)


def _raw_moment_from_quad(logpdf_fn, params: tuple[float, ...], order: int) -> float:
    def integrand(x: float) -> float:
        return float((x**order) * np.exp(logpdf_fn(np.array([x]), *params)[0]))

    left, _ = quad(integrand, -np.inf, 0.0, epsabs=1e-10, epsrel=1e-10, limit=200)
    right, _ = quad(integrand, 0.0, np.inf, epsabs=1e-10, epsrel=1e-10, limit=200)
    return float(left + right)


def _central_moment_from_quad(
    logpdf_fn,
    params: tuple[float, ...],
    mean: float,
    order: int,
) -> float:
    def integrand(x: float) -> float:
        return float(((x - mean) ** order) * np.exp(logpdf_fn(np.array([x]), *params)[0]))

    left, _ = quad(integrand, -np.inf, 0.0, epsabs=1e-10, epsrel=1e-10, limit=200)
    right, _ = quad(integrand, 0.0, np.inf, epsabs=1e-10, epsrel=1e-10, limit=200)
    return float(left + right)
