"""Command-line interface for the fixed scenario fitting experiment.

(layer: io) argparse-driven entry point; defers all numerical work to the
orchestration layer. A Rust port of this file would use ``clap`` and call
into the same orchestration types.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import ExperimentConfig, best_models_by_bic, run_fixed_experiment


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the fixed experiment CLI."""

    parser = argparse.ArgumentParser(
        description="Fit skewed/bimodal distributions to fixed synthetic scenarios."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for CSV, Markdown, and PNG outputs.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=800,
        help="Number of observations per scenario.",
    )
    parser.add_argument("--seed", type=int, default=20260503, help="Base random seed.")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=250,
        help="Maximum L-BFGS-B/EM iterations per fit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the fixed scenario fitting experiment."""

    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir: Path = args.output_dir
    sample_size: int = args.sample_size
    seed: int = args.seed
    max_iter: int = args.max_iter
    config = ExperimentConfig(
        output_dir=output_dir,
        sample_size=sample_size,
        seed=seed,
        max_iter=max_iter,
    )
    summary = run_fixed_experiment(config)
    best = best_models_by_bic(summary)
    print("Best model per fixed scenario by BIC:")
    print(best.to_string(index=False))
    print(f"\nWrote: {config.output_dir / 'fit_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
