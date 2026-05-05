from __future__ import annotations

import numpy as np

from bimodal_skewfit.fit import (
    fit_abn,
    fit_adn,
    fit_all,
    fit_bsn_fs,
    fit_gmm2,
    fit_normal,
    fit_ntpn,
)
from bimodal_skewfit.simulate import symmetric_bimodal


def test_individual_fitters_return_finite_nll() -> None:
    rng = np.random.default_rng(42)
    x = symmetric_bimodal(180, rng)
    results = [
        fit_normal(x),
        fit_gmm2(x, seed=42, n_random_starts=2, max_iter=80),
        fit_abn(x, max_iter=80),
        fit_adn(x, max_iter=80),
        fit_bsn_fs(x, max_iter=80),
        fit_ntpn(x, max_iter=80),
    ]
    for result in results:
        assert np.isfinite(result.nll)
        assert result.nobs == x.size
        assert len(result.params) == result.n_parameters


def test_fit_all_sorts_by_bic_and_gmm_beats_normal_on_clear_bimodal() -> None:
    rng = np.random.default_rng(123)
    x = symmetric_bimodal(240, rng)
    results = fit_all(x, seed=123, max_iter=80)
    bics = [result.bic for result in results]
    assert bics == sorted(bics)
    by_name = {result.model: result for result in results}
    assert by_name["gmm2"].bic < by_name["normal"].bic
