"""Random distribution factories for stress-testing the fitters.

(layer: driver) Per-trial random density generation. Each builder consumes a
project-supplied :class:`numpy.random.Generator`, draws parameters and a
sample, and returns a :class:`GeneratedDensity` carrying both the sample and
the closed-form pdf for ISE scoring. No I/O.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr
from scipy.stats import skewnorm

from .distributions import (
    FloatArray,
    abn_logpdf,
    adn_logpdf,
    bsn_fs_logpdf,
    gmm2_logpdf,
    normal_logpdf,
    ntpn_logpdf,
)

DensityFn = Callable[[ArrayLike], FloatArray]


@dataclass(frozen=True, slots=True)
class GeneratedDensity:
    """A generated sample plus its data-generating density for quality scoring."""

    trial_id: int
    family: str
    description: str
    params: dict[str, float]
    sample: FloatArray
    pdf: DensityFn
    sampler_diagnostics: dict[str, float] | None = None
    """Optional sampler-quality metrics (e.g. M-H acceptance rate). ``None`` for
    direct samplers that have nothing to report."""


def _sample_gmm2(n: int, rng: np.random.Generator, params: dict[str, float]) -> FloatArray:
    labels = rng.binomial(1, params["weight"], size=n)
    means = np.where(labels == 1, params["mu1"], params["mu2"])
    scales = np.where(labels == 1, params["sigma1"], params["sigma2"])
    return rng.normal(means, scales).astype(np.float64)


def _sample_abn(n: int, rng: np.random.Generator, params: dict[str, float]) -> FloatArray:
    loc = params["loc"]
    scale = params["scale"]
    lam = params["lambda"]
    alpha = params["alpha"]
    weight = 0.5 * (1.0 + alpha)
    labels = rng.binomial(1, weight, size=n)
    z_means = np.where(labels == 1, lam, -lam)
    z = rng.normal(z_means, 1.0)
    return (loc + scale * z).astype(np.float64)


def _sample_adn(n: int, rng: np.random.Generator, params: dict[str, float]) -> FloatArray:
    loc = params["loc"]
    scale = params["scale"]
    sep = params["sep"]
    skew = params["skew"]
    accepted: list[FloatArray] = []
    needed = n

    while needed > 0:
        proposal_size = max(4 * needed, 128)
        signs = rng.choice(np.array([-1.0, 1.0]), size=proposal_size)
        z = rng.normal(loc=signs * sep, scale=1.0, size=proposal_size)
        keep = rng.uniform(size=proposal_size) <= ndtr(skew * z)
        accepted_block = z[keep]
        if accepted_block.size > 0:
            accepted.append(accepted_block[:needed].astype(np.float64))
            needed -= min(needed, accepted_block.size)

    z_all = np.concatenate(accepted)[:n]
    return (loc + scale * z_all).astype(np.float64)


def _normal_density(trial_id: int, n: int, rng: np.random.Generator) -> GeneratedDensity:
    loc = float(rng.uniform(-1.5, 1.5))
    scale = float(rng.uniform(0.55, 1.8))
    sample = rng.normal(loc=loc, scale=scale, size=n).astype(np.float64)

    def pdf(x: ArrayLike) -> FloatArray:
        return np.exp(normal_logpdf(x, loc, scale))

    return GeneratedDensity(
        trial_id=trial_id,
        family="normal",
        description="random single Gaussian",
        params={"loc": loc, "scale": scale},
        sample=sample,
        pdf=pdf,
    )


def _skewnorm_density(trial_id: int, n: int, rng: np.random.Generator) -> GeneratedDensity:
    shape = float(rng.uniform(-8.0, 8.0))
    loc = float(rng.uniform(-1.0, 1.0))
    scale = float(rng.uniform(0.55, 1.6))
    values = skewnorm.rvs(a=shape, loc=loc, scale=scale, size=n, random_state=rng)
    sample = np.asarray(values, dtype=np.float64)

    def pdf(x: ArrayLike) -> FloatArray:
        return np.asarray(skewnorm.pdf(x, a=shape, loc=loc, scale=scale), dtype=np.float64)

    return GeneratedDensity(
        trial_id=trial_id,
        family="skewnorm",
        description="random skew-normal unimodal density",
        params={"shape": shape, "loc": loc, "scale": scale},
        sample=sample,
        pdf=pdf,
    )


def _gmm2_density(trial_id: int, n: int, rng: np.random.Generator) -> GeneratedDensity:
    weight = float(rng.uniform(0.15, 0.85))
    center = float(rng.uniform(-0.7, 0.7))
    separation = float(rng.uniform(0.7, 4.5))
    mu1 = center - separation * float(rng.uniform(0.35, 0.65))
    mu2 = center + separation * float(rng.uniform(0.35, 0.65))
    sigma1 = float(rng.uniform(0.35, 1.25))
    sigma2 = float(rng.uniform(0.35, 1.35))
    params = {
        "weight": weight,
        "mu1": mu1,
        "sigma1": sigma1,
        "mu2": mu2,
        "sigma2": sigma2,
    }
    sample = _sample_gmm2(n, rng, params)

    def pdf(x: ArrayLike) -> FloatArray:
        return np.exp(gmm2_logpdf(x, weight, mu1, sigma1, mu2, sigma2))

    return GeneratedDensity(
        trial_id=trial_id,
        family="gmm2",
        description="random K=2 Gaussian mixture",
        params=params,
        sample=sample,
        pdf=pdf,
    )


def _abn_density(trial_id: int, n: int, rng: np.random.Generator) -> GeneratedDensity:
    loc = float(rng.uniform(-1.0, 1.0))
    scale = float(rng.uniform(0.55, 1.55))
    lam = float(rng.uniform(0.25, 2.8))
    alpha = float(rng.uniform(-0.85, 0.85))
    params = {"loc": loc, "scale": scale, "lambda": lam, "alpha": alpha}
    sample = _sample_abn(n, rng, params)

    def pdf(x: ArrayLike) -> FloatArray:
        return np.exp(abn_logpdf(x, loc, scale, lam, alpha))

    return GeneratedDensity(
        trial_id=trial_id,
        family="abn",
        description="random constrained asymmetric bimodal normal",
        params=params,
        sample=sample,
        pdf=pdf,
    )


def _adn_density(trial_id: int, n: int, rng: np.random.Generator) -> GeneratedDensity:
    loc = float(rng.uniform(-1.0, 1.0))
    scale = float(rng.uniform(0.55, 1.55))
    sep = float(rng.uniform(0.0, 2.6))
    skew = float(rng.uniform(-5.0, 5.0))
    params = {"loc": loc, "scale": scale, "sep": sep, "skew": skew}
    sample = _sample_adn(n, rng, params)

    def pdf(x: ArrayLike) -> FloatArray:
        return np.exp(adn_logpdf(x, loc, scale, sep, skew))

    return GeneratedDensity(
        trial_id=trial_id,
        family="adn",
        description="random asymmetric double normal",
        params=params,
        sample=sample,
        pdf=pdf,
    )


LogPdfArray = Callable[[FloatArray], FloatArray]
ProposalSampler = Callable[[int, np.random.Generator], FloatArray]


def _independent_mh_sample(
    n: int,
    target_logpdf: LogPdfArray,
    proposal_sampler: ProposalSampler,
    proposal_logpdf: LogPdfArray,
    rng: np.random.Generator,
    burnin: int = 1000,
    thin: int = 5,
) -> tuple[FloatArray, float]:
    """Independent Metropolis-Hastings with vectorised proposal evaluation.

    Returns the thinned post-burnin chain together with the empirical
    acceptance rate over all (burnin + n * thin) - 1 transitions. The
    acceptance rate is exposed for sampler-quality auditing — for a target
    whose tails are much heavier than the proposal it can drop close to
    zero, which signals that the chain is stuck and the resulting samples
    are biased.

    The acceptance ratio reduces to ``log_target(x_prop) - log_proposal(x_prop)``
    minus the same quantity at the current state, which we precompute once for
    all proposals. The chain itself runs in a Python loop because the
    accept/reject decision is path-dependent.
    """

    total = burnin + n * thin
    proposed = proposal_sampler(total, rng)
    log_target = target_logpdf(proposed)
    log_proposal = proposal_logpdf(proposed)
    log_weights = log_target - log_proposal
    log_uniform = np.log(rng.uniform(size=total))

    chain = np.empty(total, dtype=np.float64)
    chain[0] = proposed[0]
    current_weight = float(log_weights[0])
    accepted = 0

    for t in range(1, total):
        log_alpha = float(log_weights[t]) - current_weight
        if log_uniform[t] < log_alpha:
            chain[t] = proposed[t]
            current_weight = float(log_weights[t])
            accepted += 1
        else:
            chain[t] = chain[t - 1]

    acceptance_rate = accepted / float(total - 1) if total > 1 else 0.0
    return chain[burnin::thin][:n], acceptance_rate


def _propose_ntpn_base(loc: float, scale: float, lam: float) -> tuple[ProposalSampler, LogPdfArray]:
    """Build sampler/logpdf for the NTPN proposal: TPN with the same lambda.

    The proposal is the alpha=0 special case of NTPN (the tilt collapses to 1),
    so its log density is exactly ``ntpn_logpdf(x, loc, scale, 0.0, lam)``.
    """

    def sampler(n: int, rng: np.random.Generator) -> FloatArray:
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        z = rng.normal(loc=signs * lam, scale=1.0, size=n)
        return (loc + scale * z).astype(np.float64)

    def logpdf(x: FloatArray) -> FloatArray:
        return ntpn_logpdf(x, loc, scale, 0.0, lam)

    return sampler, logpdf


def _propose_bsn_fs_base(
    loc: float, scale: float, gamma: float
) -> tuple[ProposalSampler, LogPdfArray]:
    """Build sampler/logpdf for the BSN-FS proposal: Fernandez-Steel skew normal.

    Setting alpha=0 in :func:`bsn_fs_logpdf` yields exactly the FS skew normal
    base, so we reuse the kernel for the proposal log density.
    """

    inv_gamma = 1.0 / gamma
    p_positive = gamma / (gamma + inv_gamma)

    def sampler(n: int, rng: np.random.Generator) -> FloatArray:
        is_positive = rng.uniform(size=n) < p_positive
        half_normal = np.abs(rng.standard_normal(size=n))
        z = np.where(is_positive, half_normal * gamma, -half_normal * inv_gamma)
        return (loc + scale * z).astype(np.float64)

    def logpdf(x: FloatArray) -> FloatArray:
        return bsn_fs_logpdf(x, loc, scale, 0.0, gamma)

    return sampler, logpdf


def _ntpn_density(trial_id: int, n: int, rng: np.random.Generator) -> GeneratedDensity:
    loc = float(rng.uniform(-1.0, 1.0))
    scale = float(rng.uniform(0.55, 1.55))
    alpha = float(rng.uniform(-3.0, 3.0))
    lam = float(rng.uniform(0.1, 2.5))
    params = {"loc": loc, "scale": scale, "alpha": alpha, "lambda": lam}

    proposal_sampler, proposal_logpdf = _propose_ntpn_base(loc, scale, lam)

    def target_logpdf(x: FloatArray) -> FloatArray:
        return ntpn_logpdf(x, loc, scale, alpha, lam)

    sample, acceptance_rate = _independent_mh_sample(
        n, target_logpdf, proposal_sampler, proposal_logpdf, rng
    )

    def pdf(x: ArrayLike) -> FloatArray:
        return np.exp(ntpn_logpdf(x, loc, scale, alpha, lam))

    return GeneratedDensity(
        trial_id=trial_id,
        family="ntpn",
        description="random NTPN bimodal skew",
        params=params,
        sample=sample,
        pdf=pdf,
        sampler_diagnostics={"mh_acceptance_rate": acceptance_rate},
    )


def _bsn_fs_density(trial_id: int, n: int, rng: np.random.Generator) -> GeneratedDensity:
    loc = float(rng.uniform(-1.0, 1.0))
    scale = float(rng.uniform(0.55, 1.55))
    alpha = float(rng.uniform(0.0, 5.0))
    gamma = float(rng.uniform(0.5, 2.0))
    params = {"loc": loc, "scale": scale, "alpha": alpha, "gamma": gamma}

    proposal_sampler, proposal_logpdf = _propose_bsn_fs_base(loc, scale, gamma)

    def target_logpdf(x: FloatArray) -> FloatArray:
        return bsn_fs_logpdf(x, loc, scale, alpha, gamma)

    sample, acceptance_rate = _independent_mh_sample(
        n, target_logpdf, proposal_sampler, proposal_logpdf, rng
    )

    def pdf(x: ArrayLike) -> FloatArray:
        return np.exp(bsn_fs_logpdf(x, loc, scale, alpha, gamma))

    return GeneratedDensity(
        trial_id=trial_id,
        family="bsn_fs",
        description="random Fernandez-Steel bimodal skew normal",
        params=params,
        sample=sample,
        pdf=pdf,
        sampler_diagnostics={"mh_acceptance_rate": acceptance_rate},
    )


DensityBuilder = Callable[[int, int, np.random.Generator], GeneratedDensity]

RANDOM_DENSITY_BUILDERS: tuple[DensityBuilder, ...] = (
    _normal_density,
    _skewnorm_density,
    _gmm2_density,
    _abn_density,
    _adn_density,
    _ntpn_density,
    _bsn_fs_density,
)
"""Ordered registry of every random-density factory used by the quality search.

Exposed as a module-level constant so audits can verify family coverage
without re-running the stochastic ``generate_random_density`` and inspecting
its output. The ordering also fixes the integer index that ``rng.integers``
draws against, so changes here propagate into the random-search snapshot.
"""

RANDOM_DENSITY_FAMILIES: frozenset[str] = frozenset(
    {
        "normal",
        "skewnorm",
        "gmm2",
        "abn",
        "adn",
        "ntpn",
        "bsn_fs",
    }
)
"""Family name set produced by :data:`RANDOM_DENSITY_BUILDERS`."""


def generate_random_density(trial_id: int, n: int, rng: np.random.Generator) -> GeneratedDensity:
    """Draw one random density family and sample from it."""

    builder = RANDOM_DENSITY_BUILDERS[int(rng.integers(0, len(RANDOM_DENSITY_BUILDERS)))]
    return builder(trial_id, n, rng)
