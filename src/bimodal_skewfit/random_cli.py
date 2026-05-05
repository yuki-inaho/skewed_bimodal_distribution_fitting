"""Command-line interface for randomized quality-search experiments.

(layer: io) argparse-driven entry point for the randomized pipeline. As with
:mod:`cli`, all numerical work is delegated to the orchestration layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .random_search import RandomSearchConfig, run_random_search, summarize_best_quality


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the randomized search CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Randomly generate distributions, fit candidate models, and report best quality."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--sample-size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--max-iter", type=int, default=55)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run randomized generation, fitting, scoring, and reporting."""

    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir: Path = args.output_dir
    trials: int = args.trials
    sample_size: int = args.sample_size
    seed: int = args.seed
    max_iter: int = args.max_iter
    config = RandomSearchConfig(
        output_dir=output_dir,
        trials=trials,
        sample_size=sample_size,
        seed=seed,
        max_iter=max_iter,
    )
    summary = run_random_search(config)
    best = summarize_best_quality(summary)
    trial_id = int(best["trial_id"])
    true_family = str(best["true_family"])
    model = str(best["model"])
    score = float(best["quality_score"])
    ise = float(best["ise_to_true_pdf"])
    best_bic = float(best["bic"])
    print("Best randomized fitting-quality result:")
    print(
        f"trial={trial_id}, true_family={true_family}, "
        f"model={model}, score={score:.3f}, "
        f"ISE={ise:.6f}, BIC={best_bic:.3f}"
    )
    print(f"\nWrote: {config.output_dir / 'random_search_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
