"""Regression tests that lock numerical and structural outputs in place.

These tests are intentionally narrow: they execute the fixed-scenario and
randomized search pipelines on small, deterministic configurations and compare
against snapshot values produced on a known-good run. Loose tolerances are used
on optimizer-driven values so that minor floating-point drift between equivalent
scipy/numpy patch releases does not cause false failures, while the exact best
model selection and output schema are pinned.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from bimodal_skewfit.distributions import location_scale_z
from bimodal_skewfit.experiment import (
    ExperimentConfig,
    best_models_by_bic,
    run_fixed_experiment,
)
from bimodal_skewfit.fit import fit_normal
from bimodal_skewfit.random_search import (
    RandomSearchConfig,
    run_random_search,
    summarize_best_quality,
)
from bimodal_skewfit.simulate import SCENARIOS, generate_scenario_data

# Pinned experiment configurations. Reference values below are tied to these.
FIXED_SAMPLE_SIZE = 200
FIXED_SEED = 20260503
FIXED_MAX_ITER = 40

RANDOM_TRIALS = 4
RANDOM_SAMPLE_SIZE = 120
RANDOM_SEED = 20260504
RANDOM_MAX_ITER = 25

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Data reproducibility
# ---------------------------------------------------------------------------

LOCKED_SCENARIO_HEAD = {
    "normal_unimodal": [1.013431, 0.912517, 1.794163],
    "skewed_unimodal": [0.648376, 0.717783, 1.558702],
    "symmetric_bimodal": [3.014765, -1.388421, 1.845044],
    "skewed_bimodal_mixture": [2.158647, 1.81749, -2.402635],
    "overlapping_gmm2": [0.058647, -0.28251, 1.337766],
}

LOCKED_SCENARIO_MOMENTS = {
    "normal_unimodal": (0.100943, 0.940616),
    "skewed_unimodal": (0.505363, 0.699207),
    "symmetric_bimodal": (0.187656, 2.053434),
    "skewed_bimodal_mixture": (0.123812, 1.794090),
    "overlapping_gmm2": (0.095381, 1.320452),
}


@pytest.mark.parametrize("name", list(LOCKED_SCENARIO_HEAD.keys()))
def test_generate_scenario_data_locks_first_values(name: str) -> None:
    sample = generate_scenario_data(name, FIXED_SAMPLE_SIZE, FIXED_SEED)
    np.testing.assert_allclose(sample[:3], LOCKED_SCENARIO_HEAD[name], atol=1e-6)


@pytest.mark.parametrize("name", list(LOCKED_SCENARIO_MOMENTS.keys()))
def test_generate_scenario_data_locks_sample_moments(name: str) -> None:
    sample = generate_scenario_data(name, FIXED_SAMPLE_SIZE, FIXED_SEED)
    expected_mean, expected_std = LOCKED_SCENARIO_MOMENTS[name]
    assert sample.shape == (FIXED_SAMPLE_SIZE,)
    assert sample.dtype == np.float64
    assert float(np.mean(sample)) == pytest.approx(expected_mean, abs=1e-6)
    assert float(np.std(sample)) == pytest.approx(expected_std, abs=1e-6)


def test_generate_scenario_data_is_repeatable_across_calls() -> None:
    a = generate_scenario_data("symmetric_bimodal", 256, 20260503)
    b = generate_scenario_data("symmetric_bimodal", 256, 20260503)
    np.testing.assert_array_equal(a, b)


def test_generate_scenario_data_rejects_unknown_name() -> None:
    with pytest.raises(KeyError):
        generate_scenario_data("does_not_exist", 16, 0)


# ---------------------------------------------------------------------------
# Closed-form fit
# ---------------------------------------------------------------------------


def test_fit_normal_matches_closed_form_on_normal_sample() -> None:
    sample = generate_scenario_data("normal_unimodal", FIXED_SAMPLE_SIZE, FIXED_SEED)
    result = fit_normal(sample)
    assert result.model == "normal"
    assert result.params["loc"] == pytest.approx(float(np.mean(sample)), abs=1e-12)
    assert result.params["scale"] == pytest.approx(float(np.std(sample, ddof=0)), abs=1e-12)
    assert result.n_parameters == 2
    assert result.nobs == FIXED_SAMPLE_SIZE
    assert result.converged is True
    # AIC / BIC consistency: AIC = 2k + 2*nll, BIC = k*log(n) + 2*nll.
    expected_aic = 2 * result.n_parameters + 2 * result.nll
    expected_bic = result.n_parameters * np.log(result.nobs) + 2 * result.nll
    assert result.aic == pytest.approx(expected_aic, rel=1e-12)
    assert result.bic == pytest.approx(expected_bic, rel=1e-12)


# ---------------------------------------------------------------------------
# Distribution boundary handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_scale", [0.0, -1.5, float("nan"), float("inf")])
def test_location_scale_z_rejects_nonpositive_or_nonfinite_scale(bad_scale: float) -> None:
    with pytest.raises(ValueError):
        location_scale_z(np.array([0.0, 1.0]), 0.0, bad_scale)


# ---------------------------------------------------------------------------
# Fixed-scenario experiment regression
# ---------------------------------------------------------------------------

LOCKED_BEST_PER_SCENARIO = {
    "normal_unimodal": {"model": "normal", "bic": 553.683928, "mode_count_grid": 1},
    "skewed_unimodal": {"model": "bsn_fs", "bic": 434.164136, "mode_count_grid": 1},
    "symmetric_bimodal": {"model": "ntpn", "bic": 664.997373, "mode_count_grid": 2},
    "skewed_bimodal_mixture": {"model": "gmm2", "bic": 659.276523, "mode_count_grid": 2},
    "overlapping_gmm2": {"model": "gmm2", "bic": 637.579628, "mode_count_grid": 2},
}

EXPECTED_FIT_SUMMARY_COLUMNS = {
    "scenario",
    "scenario_description",
    "model",
    "nll",
    "aic",
    "bic",
    "loglik",
    "converged",
    "mode_count_grid",
    "message",
}


@pytest.fixture(scope="module")
def fixed_experiment(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, pd.DataFrame]:
    output_dir = tmp_path_factory.mktemp("fixed_experiment")
    cfg = ExperimentConfig(
        output_dir=output_dir,
        sample_size=FIXED_SAMPLE_SIZE,
        seed=FIXED_SEED,
        max_iter=FIXED_MAX_ITER,
    )
    summary = run_fixed_experiment(cfg)
    return output_dir, summary


def test_fixed_experiment_summary_has_expected_columns(
    fixed_experiment: tuple[Path, pd.DataFrame],
) -> None:
    _, summary = fixed_experiment
    assert EXPECTED_FIT_SUMMARY_COLUMNS.issubset(summary.columns)
    assert summary["model"].isin({"normal", "gmm2", "abn", "adn", "bsn_fs", "ntpn"}).all()
    assert summary["scenario"].nunique() == len(SCENARIOS)
    assert len(summary) == len(SCENARIOS) * 6


def test_fixed_experiment_finite_objectives_and_best_model_converged(
    fixed_experiment: tuple[Path, pd.DataFrame],
) -> None:
    _, summary = fixed_experiment
    assert np.isfinite(summary["nll"]).all()
    assert np.isfinite(summary["bic"]).all()
    # The best (BIC-minimising) model per scenario must converge. Misspecified
    # candidates may legitimately fail to converge under a tight max_iter; this
    # is fine as long as the winner did.
    best_indices = summary.groupby("scenario")["bic"].idxmin()
    best_rows = summary.loc[best_indices]
    assert best_rows["converged"].all(), (
        f"best-by-BIC model failed to converge: {best_rows[~best_rows['converged']]}"
    )


@pytest.mark.parametrize("scenario", list(LOCKED_BEST_PER_SCENARIO.keys()))
def test_fixed_experiment_best_model_is_locked(
    fixed_experiment: tuple[Path, pd.DataFrame], scenario: str
) -> None:
    _, summary = fixed_experiment
    best = best_models_by_bic(summary).set_index("scenario").loc[scenario]
    expected = LOCKED_BEST_PER_SCENARIO[scenario]
    assert best["model"] == expected["model"]
    assert int(best["mode_count_grid"]) == expected["mode_count_grid"]
    assert float(best["bic"]) == pytest.approx(expected["bic"], rel=1e-4)


def test_fixed_experiment_writes_summary_csv_and_png_overlays(
    fixed_experiment: tuple[Path, pd.DataFrame],
) -> None:
    output_dir, summary = fixed_experiment
    summary_path = output_dir / "fit_summary.csv"
    assert summary_path.exists()
    reloaded = pd.read_csv(summary_path)
    assert len(reloaded) == len(summary)

    for scenario in SCENARIOS:
        png_path = output_dir / f"{scenario.name}_fit_overlay.png"
        assert png_path.exists(), f"missing PNG for scenario={scenario.name}"
        assert png_path.stat().st_size > 0
        with png_path.open("rb") as handle:
            assert handle.read(8) == PNG_SIGNATURE


# ---------------------------------------------------------------------------
# Randomized search regression
# ---------------------------------------------------------------------------

EXPECTED_RANDOM_SUMMARY_COLUMNS = {
    "trial_id",
    "true_family",
    "true_description",
    "model",
    "bic_rank",
    "nll",
    "aic",
    "bic",
    "loglik",
    "converged",
    "true_mode_count",
    "fitted_mode_count",
    "mode_match",
    "ise_to_true_pdf",
    "bic_gap_to_best",
    "bic_delta_best_second",
    "quality_score",
    "message",
}

LOCKED_RANDOM_BEST = {
    "trial_id": 2,
    "true_family": "gmm2",
    "model": "abn",
    "quality_score": 131.798009,
    "ise_to_true_pdf": 0.00020097,
    "bic": 447.405199,
    "mode_match": True,
}


@pytest.fixture(scope="module")
def random_search_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, pd.DataFrame]:
    output_dir = tmp_path_factory.mktemp("random_search")
    cfg = RandomSearchConfig(
        output_dir=output_dir,
        trials=RANDOM_TRIALS,
        sample_size=RANDOM_SAMPLE_SIZE,
        seed=RANDOM_SEED,
        max_iter=RANDOM_MAX_ITER,
    )
    summary = run_random_search(cfg)
    return output_dir, summary


def test_random_search_summary_has_expected_columns(
    random_search_run: tuple[Path, pd.DataFrame],
) -> None:
    _, summary = random_search_run
    assert EXPECTED_RANDOM_SUMMARY_COLUMNS.issubset(summary.columns)
    assert len(summary) == RANDOM_TRIALS * 6
    assert summary["trial_id"].nunique() == RANDOM_TRIALS
    assert (summary["ise_to_true_pdf"] >= 0.0).all()
    assert summary["bic_rank"].between(1, 6).all()


def test_random_search_writes_outputs(
    random_search_run: tuple[Path, pd.DataFrame],
) -> None:
    output_dir, _ = random_search_run
    assert (output_dir / "random_search_summary.csv").exists()
    assert (output_dir / "random_quality_report.md").exists()
    best_png = output_dir / "random_search_best_fit.png"
    assert best_png.exists()
    with best_png.open("rb") as handle:
        assert handle.read(8) == PNG_SIGNATURE


def test_random_search_best_quality_result_is_locked(
    random_search_run: tuple[Path, pd.DataFrame],
) -> None:
    _, summary = random_search_run
    best = summarize_best_quality(summary)
    assert int(best["trial_id"]) == LOCKED_RANDOM_BEST["trial_id"]
    assert best["true_family"] == LOCKED_RANDOM_BEST["true_family"]
    assert best["model"] == LOCKED_RANDOM_BEST["model"]
    assert bool(best["mode_match"]) is LOCKED_RANDOM_BEST["mode_match"]
    assert float(best["quality_score"]) == pytest.approx(
        LOCKED_RANDOM_BEST["quality_score"], rel=1e-4
    )
    assert float(best["ise_to_true_pdf"]) == pytest.approx(
        LOCKED_RANDOM_BEST["ise_to_true_pdf"], rel=5e-4
    )
    assert float(best["bic"]) == pytest.approx(LOCKED_RANDOM_BEST["bic"], rel=1e-4)


# ---------------------------------------------------------------------------
# Plotting backend isolation and notebook builder structure
# ---------------------------------------------------------------------------


def test_plotting_module_does_not_override_user_backend() -> None:
    """Importing :mod:`plotting` must not change the active matplotlib backend.

    The package is intended for interactive use; silently switching the
    backend on import would break a user's Qt5Agg / nbagg configuration.
    """

    backend_before = matplotlib.get_backend()
    import importlib

    from bimodal_skewfit import plotting

    importlib.reload(plotting)
    assert matplotlib.get_backend() == backend_before


def test_build_report_notebook_structure() -> None:
    import sys

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import build_report_notebook

    notebook = build_report_notebook.build_notebook()
    assert notebook.metadata["kernelspec"]["name"] == "python3"
    assert notebook.metadata["language_info"]["name"] == "python"
    cell_types = [cell.cell_type for cell in notebook.cells]
    assert cell_types.count("code") >= 4
    assert cell_types.count("markdown") >= 4
    for cell in notebook.cells:
        assert cell.source.strip() != ""
