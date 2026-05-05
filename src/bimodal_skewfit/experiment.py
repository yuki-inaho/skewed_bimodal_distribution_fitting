"""Reusable orchestration for fixed scenario fitting experiments.

(layer: orchestration) Coordinates :mod:`simulate` → :mod:`fit` →
:mod:`evaluate` → :mod:`plotting`, then writes a CSV summary. Owns the I/O
boundary at the experiment level; the kernel and driver layers below stay
side-effect-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import count_density_modes, empirical_quantile_grid
from .fit import FitResult, fit_all
from .plotting import plot_fit_overlay
from .simulate import SCENARIOS, Scenario


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration for the fixed scenario experiment."""

    output_dir: Path
    sample_size: int = 800
    seed: int = 20260503
    max_iter: int = 250


def result_to_row(scenario: Scenario, result: FitResult, mode_count: int) -> dict[str, object]:
    """Convert one fitted result into a flat table row."""

    row: dict[str, object] = {
        "scenario": scenario.name,
        "scenario_description": scenario.description,
        "model": result.model,
        "nll": result.nll,
        "aic": result.aic,
        "bic": result.bic,
        "loglik": result.loglik,
        "converged": result.converged,
        "mode_count_grid": mode_count,
        "message": result.message,
    }
    for key, value in result.params.items():
        row[f"param_{key}"] = value
    return row


def run_fixed_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """Run all fixed scenarios and write summary CSV plus overlay plots."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for scenario_index, scenario in enumerate(SCENARIOS):
        data_seed = config.seed + 1000 * scenario_index
        rng = np.random.default_rng(data_seed)
        data = scenario.generator(config.sample_size, rng)
        results = fit_all(data, seed=config.seed + scenario_index, max_iter=config.max_iter)
        grid = empirical_quantile_grid(data)

        for result in results:
            density = result.pdf(grid)
            mode_count = count_density_modes(grid, density)
            rows.append(result_to_row(scenario, result, mode_count))

        plot_fit_overlay(
            data,
            results,
            config.output_dir / f"{scenario.name}_fit_overlay.png",
            title=f"{scenario.name}: {scenario.description}",
        )

    summary = pd.DataFrame(rows).sort_values(["scenario", "bic", "model"])
    summary.to_csv(config.output_dir / "fit_summary.csv", index=False)
    return summary


def best_models_by_bic(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the best model per scenario according to BIC."""

    best_indices = summary.groupby("scenario")["bic"].idxmin()
    columns = ["scenario", "model", "bic", "mode_count_grid"]
    return summary.loc[best_indices, columns].sort_values("scenario")
