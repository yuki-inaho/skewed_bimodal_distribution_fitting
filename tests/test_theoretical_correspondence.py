"""Property-based tests asserting that each model density reduces to its known
limiting forms, and that the random generators / fitters expose the full
catalogue of fitted shape families.

These checks are independent of the regression snapshot tests: they confirm
that the implementation matches the published special-case identities for
ABN, ADN, BSN-FS, and NTPN, and that operational invariants (full coverage of
shape families in the random search, convergence-flag reporting in
``fit_gmm2``) hold.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import norm, skewnorm

from bimodal_skewfit.distributions import (
    LOG_SQRT_2PI,
    LOG_TWO,
    abn_logpdf,
    adn_logpdf,
    bsn_fs_b_gamma,
    bsn_fs_logpdf,
    gmm2_logpdf,
    log_phi,
    normal_logpdf,
    ntpn_logpdf,
)
from bimodal_skewfit.fit import fit_gmm2
from bimodal_skewfit.random_generators import (
    RANDOM_DENSITY_BUILDERS,
    RANDOM_DENSITY_FAMILIES,
    generate_random_density,
)

# ---------------------------------------------------------------------------
# ADN special cases
# ---------------------------------------------------------------------------


def test_adn_skew_zero_equals_equal_weight_normal_mixture() -> None:
    """ADN with skew=0 reduces to an equal-weight 2-Gaussian mixture in z."""

    x = np.linspace(-5.0, 5.0, 401)
    sep = 1.5
    actual = np.exp(adn_logpdf(x, 0.0, 1.0, sep, 0.0))
    expected = 0.5 * (norm.pdf(x, loc=sep) + norm.pdf(x, loc=-sep))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_adn_sep_zero_equals_skew_normal() -> None:
    """ADN with sep=0 reduces to the Azzalini skew-normal density."""

    x = np.linspace(-4.0, 4.0, 201)
    skew = 1.7
    actual = np.exp(adn_logpdf(x, 0.0, 1.0, 0.0, skew))
    expected = skewnorm.pdf(x, a=skew)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# ABN special cases
# ---------------------------------------------------------------------------


def test_abn_equals_constrained_gmm2() -> None:
    """ABN(loc, scale, lambda, alpha) equals an explicit constrained GMM2."""

    x = np.linspace(-6.0, 6.0, 401)
    loc, scale, lam, alpha = 0.3, 1.2, 1.4, 0.35
    weight_pos = 0.5 * (1.0 + alpha)
    mu1 = loc + scale * lam
    mu2 = loc - scale * lam
    sigma = scale
    abn_density = np.exp(abn_logpdf(x, loc, scale, lam, alpha))
    gmm_density = np.exp(gmm2_logpdf(x, weight_pos, mu1, sigma, mu2, sigma))
    np.testing.assert_allclose(abn_density, gmm_density, atol=1e-12)


def test_abn_mean_and_variance_match_closed_form() -> None:
    """First and second central moments of ABN match the constrained-GMM2 form."""

    loc, scale, lam, alpha = 0.5, 1.3, 1.1, 0.4

    def density(t: float) -> float:
        return float(np.exp(abn_logpdf(np.array([t]), loc, scale, lam, alpha))[0])

    mean_integral, _ = quad(lambda t: t * density(t), -np.inf, np.inf, epsabs=1e-9)
    expected_mean = loc + scale * lam * alpha
    assert mean_integral == pytest.approx(expected_mean, abs=1e-6)

    second_moment, _ = quad(lambda t: t * t * density(t), -np.inf, np.inf, epsabs=1e-9)
    variance = second_moment - mean_integral**2
    expected_variance = scale**2 * (1.0 + lam**2 * (1.0 - alpha**2))
    assert variance == pytest.approx(expected_variance, abs=1e-6)


# ---------------------------------------------------------------------------
# NTPN special cases
# ---------------------------------------------------------------------------


def test_ntpn_alpha_zero_equals_two_piece_normal() -> None:
    """NTPN with alpha=0 reduces to the symmetric two-piece-normal kernel."""

    x = np.linspace(-5.0, 5.0, 401)
    lam = 1.4
    z = x  # loc=0, scale=1
    tpn = np.exp(logsumexp(np.vstack([log_phi(z - lam), log_phi(z + lam)]), axis=0) - LOG_TWO)
    actual = np.exp(ntpn_logpdf(x, 0.0, 1.0, 0.0, lam))
    np.testing.assert_allclose(actual, tpn, atol=1e-12)


def test_ntpn_lambda_zero_equals_alpha_skew_normal() -> None:
    """NTPN at lambda=0 collapses to ((1 - alpha z)^2 + 1)/(alpha^2 + 2) * phi(z)."""

    x = np.linspace(-5.0, 5.0, 401)
    alpha = 0.8
    actual = np.exp(ntpn_logpdf(x, 0.0, 1.0, alpha, 0.0))
    expected = ((1.0 - alpha * x) ** 2 + 1.0) / (alpha**2 + 2.0) * norm.pdf(x)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_ntpn_alpha_lambda_zero_equals_normal() -> None:
    """NTPN at alpha=0, lambda=0 reduces to the standard normal."""

    x = np.linspace(-4.0, 4.0, 51)
    actual = ntpn_logpdf(x, 0.0, 1.0, 0.0, 0.0)
    expected = normal_logpdf(x, 0.0, 1.0)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_ntpn_sign_reflection_identity() -> None:
    """f_ntpn(x; alpha, lambda) == f_ntpn(-x; -alpha, lambda) (TPN symmetry + tilt sign flip)."""

    x = np.linspace(-4.0, 4.0, 81)
    alpha, lam = 0.9, 1.3
    left = ntpn_logpdf(x, 0.0, 1.0, alpha, lam)
    right = ntpn_logpdf(-x, 0.0, 1.0, -alpha, lam)
    np.testing.assert_allclose(left, right, atol=1e-12)


# ---------------------------------------------------------------------------
# BSN-FS special cases
# ---------------------------------------------------------------------------


def test_bsn_fs_alpha_zero_equals_fs_skew_normal() -> None:
    """BSN-FS with alpha=0 reduces to the Fernandez-Steel skew-normal kernel."""

    x = np.linspace(-5.0, 5.0, 401)
    gamma = 1.6
    actual = np.exp(bsn_fs_logpdf(x, 0.0, 1.0, 0.0, gamma))

    log_const = np.log(2.0) - np.log(gamma + 1.0 / gamma) - LOG_SQRT_2PI
    z = x
    expected_log = np.where(
        z >= 0.0,
        log_const - 0.5 * (z / gamma) ** 2,
        log_const - 0.5 * (z * gamma) ** 2,
    )
    np.testing.assert_allclose(actual, np.exp(expected_log), atol=1e-12)


def test_bsn_fs_b_gamma_constants() -> None:
    """The b_gamma normalising constant matches the closed-form arXiv expression."""

    for gamma in (0.5, 0.8, 1.0, 1.5, 2.5):
        expected = (gamma**3 + gamma**-3) / (gamma + gamma**-1)
        assert bsn_fs_b_gamma(gamma) == pytest.approx(expected, rel=1e-15)


# ---------------------------------------------------------------------------
# Operational invariants
# ---------------------------------------------------------------------------


def test_random_density_registry_covers_all_supported_families() -> None:
    """The registry constant must enumerate every supported family exactly once.

    This is a structural assertion against the module-level constants — no
    sampling or seeded draw is involved, so the test is deterministic by
    construction. If a family is fittable but missing from the registry, the
    quality search cannot demonstrate recovery of that family.
    """

    expected_families = {"normal", "skewnorm", "gmm2", "abn", "adn", "bsn_fs", "ntpn"}
    assert expected_families == RANDOM_DENSITY_FAMILIES
    assert len(RANDOM_DENSITY_BUILDERS) == len(expected_families)


def test_random_density_builders_emit_advertised_families() -> None:
    """Each builder in the registry must produce a sample whose ``family``
    attribute matches its advertised entry in :data:`RANDOM_DENSITY_FAMILIES`.
    """

    rng = np.random.default_rng(20260601)
    seen: set[str] = set()
    for builder in RANDOM_DENSITY_BUILDERS:
        generated = builder(0, 64, rng)
        seen.add(generated.family)
    assert seen == RANDOM_DENSITY_FAMILIES


def test_generate_random_density_can_emit_every_family() -> None:
    """Smoke check that ``generate_random_density`` itself eventually visits
    every family. Uses 200 trials, which under uniform draw covers all 7
    builders with negligible miss probability.
    """

    seen: set[str] = set()
    rng = np.random.default_rng(20260602)
    for trial_id in range(200):
        generated = generate_random_density(trial_id, n=32, rng=rng)
        seen.add(generated.family)
        if seen == RANDOM_DENSITY_FAMILIES:
            break
    assert seen == RANDOM_DENSITY_FAMILIES


def test_mh_random_builders_report_reasonable_acceptance_rate() -> None:
    """Independent M-H samplers must expose a non-trivial acceptance rate.

    A degenerate proposal would drive the rate to ~0 and silently produce a
    biased sample. We require both NTPN and BSN-FS builders to expose a
    ``mh_acceptance_rate`` diagnostic and to clear a loose 5 % floor.
    """

    from bimodal_skewfit.random_generators import _bsn_fs_density, _ntpn_density

    rng = np.random.default_rng(20260605)
    for builder in (_ntpn_density, _bsn_fs_density):
        gen = builder(0, 200, rng)
        assert gen.sampler_diagnostics is not None
        rate = gen.sampler_diagnostics["mh_acceptance_rate"]
        assert 0.05 < rate < 1.0, (
            f"{gen.family}: acceptance rate {rate:.4f} is outside the audit window"
        )


def test_fit_gmm2_reports_best_run_convergence_not_any_run_convergence() -> None:
    """``fit_gmm2`` returns the *best* run's converged flag, not the OR over runs.

    With ``max_iter=1`` no EM run can satisfy the |Δ ll| < tol threshold, so the
    best run must report ``converged=False``. A bug that ORed all runs together
    would still allow ``converged=True`` if any borderline early-exit happened.
    """

    rng = np.random.default_rng(42)
    sample = np.concatenate(
        [rng.normal(-2.0, 0.6, size=120), rng.normal(2.0, 0.6, size=120)],
    )
    result = fit_gmm2(sample, seed=42, n_random_starts=4, max_iter=1, tol=1e-8)
    assert result.converged is False, (
        "fit_gmm2 reported converged=True even though every EM run was forced "
        "to exit before satisfying the tolerance"
    )
