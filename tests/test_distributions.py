from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from bimodal_skewfit.distributions import (
    FloatArray,
    abn_logpdf,
    adn_logpdf,
    bsn_fs_b_gamma,
    bsn_fs_logpdf,
    normal_logpdf,
    ntpn_logpdf,
)

LogPdfForTest = Callable[..., FloatArray]


def integrate_pdf(logpdf_fn: LogPdfForTest, *params: float) -> float:
    value, error = quad(
        lambda t: float(np.exp(logpdf_fn(np.array([t]), *params)[0])),
        -np.inf,
        np.inf,
        epsabs=2e-8,
    )
    assert error < 1e-5
    return float(value)


def test_standard_normal_density_integrates_to_one() -> None:
    assert abs(integrate_pdf(normal_logpdf, 0.0, 1.0) - 1.0) < 1e-7


def test_abn_density_integrates_to_one() -> None:
    assert abs(integrate_pdf(abn_logpdf, 0.0, 1.0, 1.4, 0.35) - 1.0) < 1e-7


def test_adn_density_integrates_to_one_and_nested_cases() -> None:
    assert abs(integrate_pdf(adn_logpdf, 0.0, 1.0, 1.1, -2.0) - 1.0) < 1e-7
    x = np.linspace(-3, 3, 11)
    skew = 1.7
    expected = 2.0 * norm.pdf(x) * norm.cdf(skew * x)
    actual = np.exp(adn_logpdf(x, 0.0, 1.0, 0.0, skew))
    assert np.allclose(actual, expected, atol=1e-12)


def test_bsn_fs_density_integrates_to_one() -> None:
    assert bsn_fs_b_gamma(1.0) == 1.0
    assert abs(integrate_pdf(bsn_fs_logpdf, 0.0, 1.0, 1.5, 1.8) - 1.0) < 1e-7


def test_ntpn_density_integrates_to_one_and_nested_normal() -> None:
    assert abs(integrate_pdf(ntpn_logpdf, 0.0, 1.0, 0.9, 1.3) - 1.0) < 1e-7
    x = np.linspace(-3, 3, 21)
    assert np.allclose(ntpn_logpdf(x, 0.0, 1.0, 0.0, 0.0), normal_logpdf(x, 0.0, 1.0))
