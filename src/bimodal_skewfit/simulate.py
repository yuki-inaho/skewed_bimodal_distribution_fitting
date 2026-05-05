"""Synthetic fixed scenarios for fitting demonstrations.

(layer: driver) Produces deterministic samples from named scenarios using a
project-supplied seed. RNG-only side effect; no file/network I/O.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import skewnorm

from .distributions import FloatArray

ScenarioGenerator = Callable[[int, np.random.Generator], FloatArray]


@dataclass(frozen=True, slots=True)
class Scenario:
    """Definition of one reproducible synthetic-data scenario."""

    name: str
    description: str
    generator: ScenarioGenerator


def normal_unimodal(n: int, rng: np.random.Generator) -> FloatArray:
    """Generate a symmetric unimodal normal sample."""

    return rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float64)


def skewed_unimodal(n: int, rng: np.random.Generator) -> FloatArray:
    """Generate a skew-normal-like unimodal sample."""

    values = skewnorm.rvs(a=6.0, loc=-0.4, scale=1.2, size=n, random_state=rng)
    return np.asarray(values, dtype=np.float64)


def symmetric_bimodal(n: int, rng: np.random.Generator) -> FloatArray:
    """Generate a well-separated equal-weight, equal-scale Gaussian mixture."""

    labels = rng.binomial(1, 0.5, size=n)
    means = np.where(labels == 1, 2.0, -2.0)
    return rng.normal(loc=means, scale=0.65, size=n).astype(np.float64)


def skewed_bimodal_mixture(n: int, rng: np.random.Generator) -> FloatArray:
    """Generate an unequal-weight, unequal-scale two-Gaussian mixture."""

    labels = rng.binomial(1, 0.72, size=n)
    means = np.where(labels == 1, 1.3, -2.2)
    scales = np.where(labels == 1, 0.55, 0.85)
    return rng.normal(loc=means, scale=scales, size=n).astype(np.float64)


def overlapping_gmm2(n: int, rng: np.random.Generator) -> FloatArray:
    """Generate an overlapping two-Gaussian mixture with unequal variances."""

    labels = rng.binomial(1, 0.35, size=n)
    means = np.where(labels == 1, 1.6, -0.8)
    scales = np.where(labels == 1, 1.1, 0.55)
    return rng.normal(loc=means, scale=scales, size=n).astype(np.float64)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("normal_unimodal", "single symmetric Gaussian", normal_unimodal),
    Scenario("skewed_unimodal", "single skew-normal-like unimodal sample", skewed_unimodal),
    Scenario(
        "symmetric_bimodal",
        "well-separated equal-weight equal-scale Gaussian mixture",
        symmetric_bimodal,
    ),
    Scenario(
        "skewed_bimodal_mixture",
        "unequal-weight unequal-scale two Gaussian mixture",
        skewed_bimodal_mixture,
    ),
    Scenario(
        "overlapping_gmm2",
        "overlapping two Gaussian mixture with unequal variances",
        overlapping_gmm2,
    ),
)


def generate_scenario_data(name: str, n: int, seed: int) -> FloatArray:
    """Generate data for one named fixed scenario."""

    rng = np.random.default_rng(seed)
    scenario_by_name = {scenario.name: scenario for scenario in SCENARIOS}
    if name not in scenario_by_name:
        raise KeyError(f"Unknown scenario: {name}")
    return scenario_by_name[name].generator(n, rng)
