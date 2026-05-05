"""Public fitting dispatch API.

(layer: driver) Thin facade over :mod:`gmm_fit` and :mod:`shape_fit`. Owns
the model-name → fitter mapping and the BIC ranking used by ``fit_all``.
Pure side-effect-free dispatch; no I/O.
"""

from __future__ import annotations

from numpy.typing import ArrayLike

from .distributions import as_float_array
from .gmm_fit import fit_gmm2, fit_normal
from .results import FitResult
from .shape_fit import fit_abn, fit_adn, fit_bsn_fs, fit_ntpn


def fit_model(model: str, x: ArrayLike, seed: int = 1234, max_iter: int = 250) -> FitResult:
    """Fit one named model with explicit dispatch."""

    if model == "normal":
        return fit_normal(x)
    if model == "gmm2":
        return fit_gmm2(x, seed=seed, max_iter=max_iter)
    if model == "abn":
        return fit_abn(x, max_iter=max_iter)
    if model == "adn":
        return fit_adn(x, max_iter=max_iter)
    if model == "bsn_fs":
        return fit_bsn_fs(x, max_iter=max_iter)
    if model == "ntpn":
        return fit_ntpn(x, max_iter=max_iter)
    raise KeyError(f"Unknown model: {model}")


def fit_all(
    x: ArrayLike,
    models: tuple[str, ...] = ("normal", "gmm2", "abn", "adn", "bsn_fs", "ntpn"),
    seed: int = 1234,
    max_iter: int = 250,
) -> list[FitResult]:
    """Fit all requested candidate models and return BIC-sorted results."""

    arr = as_float_array(x)
    results = [
        fit_model(model, arr, seed=seed + idx, max_iter=max_iter)
        for idx, model in enumerate(models)
    ]
    return sorted(results, key=lambda result: result.bic)
