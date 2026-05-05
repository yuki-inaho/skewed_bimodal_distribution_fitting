"""Utilities for fitting skewed, bimodal, and Gaussian-mixture distributions.

The package is layered into four tiers, marked in each module's docstring:

* ``kernel`` — pure math (:mod:`distributions`, :mod:`evaluate`,
  :mod:`results`, :mod:`registry`). No RNG, no I/O. Port-first targets for a
  Rust rewrite.
* ``driver`` — fitters and data factories that consume an explicit RNG seed
  but never touch disk (:mod:`fit`, :mod:`gmm_fit`, :mod:`shape_fit`,
  :mod:`simulate`, :mod:`random_generators`).
* ``orchestration`` — coordinates a full experiment and writes summary
  artifacts (:mod:`experiment`, :mod:`random_search`).
* ``io`` — argparse / matplotlib boundary (:mod:`cli`, :mod:`random_cli`,
  :mod:`plotting`).

Imports may flow only downward through these tiers.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .fit import fit_all, fit_model
from .registry import DISTRIBUTIONS
from .results import FitResult


def list_models() -> tuple[str, ...]:
    """Return the registered model names in registry order."""

    return tuple(DISTRIBUTIONS.keys())


try:
    __version__ = version("bimodal-skewfit")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"


__all__ = [
    "FitResult",
    "__version__",
    "fit_all",
    "fit_model",
    "list_models",
]
