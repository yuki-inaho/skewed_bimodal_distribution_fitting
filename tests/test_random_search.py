from __future__ import annotations

import numpy as np

from bimodal_skewfit.evaluate import count_density_modes, empirical_quantile_grid
from bimodal_skewfit.random_generators import generate_random_density
from bimodal_skewfit.random_search import RandomSearchConfig, run_random_search


def test_random_generator_exposes_finite_reference_density() -> None:
    rng = np.random.default_rng(20260504)
    generated = generate_random_density(trial_id=0, n=120, rng=rng)
    grid = empirical_quantile_grid(generated.sample)
    density = generated.pdf(grid)
    assert generated.sample.shape == (120,)
    assert np.all(np.isfinite(density))
    assert np.all(density >= 0.0)
    assert count_density_modes(grid, density) >= 1


def test_random_search_writes_summary(tmp_path) -> None:
    config = RandomSearchConfig(output_dir=tmp_path, trials=2, sample_size=80, max_iter=25)
    summary = run_random_search(config)
    assert not summary.empty
    assert (tmp_path / "random_search_summary.csv").exists()
    assert (tmp_path / "random_quality_report.md").exists()
    assert (tmp_path / "random_search_best_fit.png").exists()
