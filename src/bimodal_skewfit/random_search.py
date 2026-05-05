"""Randomized fitting-quality search across generated distributions.

(layer: orchestration) Generates random densities, fits all candidate models,
scores fits against the known generating density, then writes summary CSV,
report Markdown, and a best-fit overlay PNG. Calls into the driver layer for
all numerical work and only itself owns I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import count_density_modes, empirical_quantile_grid, integrated_squared_error
from .fit import FitResult, fit_all
from .plotting import plot_fit_overlay
from .random_generators import GeneratedDensity, generate_random_density


@dataclass(frozen=True, slots=True)
class RandomSearchConfig:
    """Configuration for randomized data generation and fitting evaluation."""

    output_dir: Path
    trials: int = 16
    sample_size: int = 320
    seed: int = 20260504
    max_iter: int = 55


def _quality_score(
    result: FitResult,
    ise: float,
    mode_match: bool,
    bic_gap_to_best: float,
) -> float:
    """Combine fit accuracy, modality agreement, convergence, and BIC parsimony."""

    mode_bonus = 20.0 if mode_match else -20.0
    convergence_bonus = 12.0 if result.converged else -12.0
    ise_penalty = min(90.0, 1000.0 * ise)
    bic_penalty = min(40.0, 0.05 * max(bic_gap_to_best, 0.0))
    return 100.0 + mode_bonus + convergence_bonus - ise_penalty - bic_penalty


def _param_columns(generated: GeneratedDensity) -> dict[str, float]:
    """Prefix data-generating parameters for tabular output."""

    return {f"true_{key}": value for key, value in generated.params.items()}


def _rows_for_trial(
    generated: GeneratedDensity,
    results: list[FitResult],
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    """Score all fitted models for one generated trial."""

    grid = empirical_quantile_grid(generated.sample, padding=0.45, num=800)
    reference_density = generated.pdf(grid)
    true_mode_count = count_density_modes(grid, reference_density)
    best_bic = min(result.bic for result in results)
    sorted_bics = sorted(result.bic for result in results)
    second_bic = sorted_bics[1] if len(sorted_bics) > 1 else sorted_bics[0]
    bic_delta_best_second = second_bic - sorted_bics[0]
    rows: list[dict[str, object]] = []

    for rank, result in enumerate(sorted(results, key=lambda item: item.bic), start=1):
        density = result.pdf(grid)
        fitted_mode_count = count_density_modes(grid, density)
        mode_match = fitted_mode_count == true_mode_count
        ise = integrated_squared_error(grid, density, reference_density)
        bic_gap_to_best = result.bic - best_bic
        row: dict[str, object] = {
            "trial_id": generated.trial_id,
            "true_family": generated.family,
            "true_description": generated.description,
            "model": result.model,
            "bic_rank": rank,
            "nll": result.nll,
            "aic": result.aic,
            "bic": result.bic,
            "loglik": result.loglik,
            "converged": result.converged,
            "true_mode_count": true_mode_count,
            "fitted_mode_count": fitted_mode_count,
            "mode_match": mode_match,
            "ise_to_true_pdf": ise,
            "bic_gap_to_best": bic_gap_to_best,
            "bic_delta_best_second": bic_delta_best_second,
            "quality_score": _quality_score(result, ise, mode_match, bic_gap_to_best),
            "message": result.message,
        }
        row.update(_param_columns(generated))
        for key, value in result.params.items():
            row[f"fit_{key}"] = value
        rows.append(row)

    return rows, grid, reference_density


def summarize_best_quality(summary: pd.DataFrame) -> pd.Series:
    """Return the highest-quality fitted row across randomized trials."""

    best_index = summary["quality_score"].idxmax()
    return summary.loc[best_index]


def _format_best_line(best: pd.Series) -> str:
    return (
        f"trial={int(best['trial_id'])}, true_family={best['true_family']}, "
        f"selected_model={best['model']}, score={best['quality_score']:.3f}, "
        f"ISE={best['ise_to_true_pdf']:.6f}, BIC={best['bic']:.3f}"
    )


def write_random_quality_report(summary: pd.DataFrame, output_path: Path) -> None:
    """Write a compact Markdown report for the randomized quality search."""

    best = summarize_best_quality(summary)
    top = summary.sort_values("quality_score", ascending=False).head(8)
    lines = [
        "# Randomized fitting-quality report",
        "",
        "## Best-quality fitted result",
        "",
        _format_best_line(best),
        "",
        "Quality score is a deterministic composite of low integrated squared error ",
        "to the known generating density, fitted/true mode-count agreement, ",
        "convergence, and BIC parsimony. Higher is better.",
        "",
        "## Top candidates",
        "",
        "| rank | trial | true family | model | score | ISE | BIC | modes true/fit |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        lines.append(
            "| "
            f"{rank} | {int(row['trial_id'])} | {row['true_family']} | {row['model']} | "
            f"{row['quality_score']:.3f} | {row['ise_to_true_pdf']:.6f} | "
            f"{row['bic']:.3f} | {int(row['true_mode_count'])}/{int(row['fitted_mode_count'])} |"
        )
    best_per_family = summary.loc[summary.groupby("true_family")["quality_score"].idxmax()]
    lines.extend(
        [
            "",
            "## Best candidate per generated family",
            "",
            "| true family | trial | model | score | ISE | BIC | modes true/fit |",
            "| --- | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in best_per_family.sort_values("true_family").iterrows():
        lines.append(
            "| "
            f"{row['true_family']} | {int(row['trial_id'])} | {row['model']} | "
            f"{row['quality_score']:.3f} | {row['ise_to_true_pdf']:.6f} | "
            f"{row['bic']:.3f} | {int(row['true_mode_count'])}/{int(row['fitted_mode_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `random_search_summary.csv`: all model-by-trial scores.",
            "- `random_search_best_fit.png`: histogram, true density, and top fitted densities.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_random_search(config: RandomSearchConfig) -> pd.DataFrame:
    """Generate random datasets, fit candidate models, score fits, and write outputs."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    all_rows: list[dict[str, object]] = []
    generated_by_id: dict[int, GeneratedDensity] = {}
    results_by_id: dict[int, list[FitResult]] = {}
    grid_by_id: dict[int, np.ndarray] = {}
    reference_by_id: dict[int, np.ndarray] = {}

    for trial_id in range(config.trials):
        generated = generate_random_density(trial_id, config.sample_size, rng)
        results = fit_all(generated.sample, seed=config.seed + trial_id, max_iter=config.max_iter)
        rows, grid, reference_density = _rows_for_trial(generated, results)
        all_rows.extend(rows)
        generated_by_id[trial_id] = generated
        results_by_id[trial_id] = results
        grid_by_id[trial_id] = grid
        reference_by_id[trial_id] = reference_density

    summary = pd.DataFrame(all_rows).sort_values(
        ["quality_score", "bic"],
        ascending=[False, True],
    )
    summary.to_csv(config.output_dir / "random_search_summary.csv", index=False)
    write_random_quality_report(summary, config.output_dir / "random_quality_report.md")

    best = summarize_best_quality(summary)
    best_trial_id = int(best["trial_id"])
    best_generated = generated_by_id[best_trial_id]
    plot_fit_overlay(
        best_generated.sample,
        results_by_id[best_trial_id],
        config.output_dir / "random_search_best_fit.png",
        title=f"Best random fit: trial {best_trial_id}, true={best_generated.family}",
        reference_grid=grid_by_id[best_trial_id],
        reference_density=reference_by_id[best_trial_id],
    )
    return summary
