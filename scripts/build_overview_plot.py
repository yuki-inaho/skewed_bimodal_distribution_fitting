"""Render a 2x2 fit-summary subplot used as the README banner.

Picks four representative scenarios from :mod:`bimodal_skewfit.simulate`,
fits all six candidate models on the same fixed seed used by
``run_experiment.py``, and overlays the top-3 BIC fits on a histogram.

Run via:

    uv run python scripts/build_overview_plot.py

Output:

    docs/fit_summary_overview.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from bimodal_skewfit.evaluate import empirical_quantile_grid
from bimodal_skewfit.fit import fit_all
from bimodal_skewfit.simulate import generate_scenario_data

SCENARIOS = (
    ("normal_unimodal", "Symmetric unimodal (Normal)"),
    ("skewed_unimodal", "Skewed unimodal (skew-normal)"),
    ("symmetric_bimodal", "Symmetric bimodal (well-separated GMM)"),
    ("skewed_bimodal_mixture", "Asymmetric bimodal (skewed GMM)"),
)
# Fixed legend order and colour mapping. Both stay constant across subplots so
# that a given model (e.g. "ntpn") is visually identifiable regardless of
# where it lands in the BIC ranking.
DISPLAY_ORDER = ("normal", "gmm2", "abn", "adn", "bsn_fs", "ntpn")
MODEL_COLORS = {
    "normal": "#1f77b4",  # blue
    "gmm2": "#ff7f0e",  # orange
    "abn": "#2ca02c",  # green
    "adn": "#d62728",  # red
    "bsn_fs": "#9467bd",  # purple
    "ntpn": "#8c564b",  # brown
}
SAMPLE_SIZE = 400
SEED = 20260503
MAX_ITER = 60
OUTPUT = Path("docs/fit_summary_overview.png")


def main() -> int:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    fig.suptitle(
        f"bimodal-skewfit: all 6 candidate models overlaid per scenario "
        f"(n={SAMPLE_SIZE}, seed={SEED}, max_iter={MAX_ITER}; best model thickened, "
        f"BIC printed in legend)",
        fontsize=11,
    )

    for ax, (name, label) in zip(axes.flat, SCENARIOS, strict=True):
        data = generate_scenario_data(name, SAMPLE_SIZE, SEED)
        grid = empirical_quantile_grid(data)
        results = fit_all(data, seed=SEED, max_iter=MAX_ITER)
        results_by_name = {r.model: r for r in results}
        best_name = min(results, key=lambda r: r.bic).model

        ax.hist(data, bins="fd", density=True, alpha=0.30, label="empirical", color="0.55")
        for model_name in DISPLAY_ORDER:
            result = results_by_name[model_name]
            is_best = model_name == best_name
            linewidth = 2.4 if is_best else 1.2
            label_suffix = " (best)" if is_best else ""
            ax.plot(
                grid,
                result.pdf(grid),
                color=MODEL_COLORS[model_name],
                linewidth=linewidth,
                label=f"{model_name}{label_suffix}  BIC={result.bic:.1f}",
            )
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.legend(loc="best", fontsize=7.5)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=140)
    plt.close(fig)
    print(f"Wrote: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
