"""Build and optionally execute a notebook summarizing fitting outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat import NotebookNode

NOTEBOOK_PATH = Path("examples/fitting_report.ipynb")


def _markdown_cell(source: str) -> NotebookNode:
    """Create one Markdown cell."""

    return nbformat.v4.new_markdown_cell(source)


def _code_cell(source: str) -> NotebookNode:
    """Create one code cell."""

    return nbformat.v4.new_code_cell(source)


def build_notebook() -> NotebookNode:
    """Create the report notebook structure."""

    cells = [
        _markdown_cell(
            "# Bimodal and skewed density fitting report\n\n"
            "This notebook reads generated CSV/PNG outputs and summarizes fixed-scenario "
            "fitting plus randomized quality-search results."
        ),
        _code_cell(
            "from pathlib import Path\n"
            "\n"
            "import pandas as pd\n"
            "from IPython.display import Image, Markdown, display\n\n"
            "ROOT = Path.cwd()\n"
            "OUTPUTS = ROOT / 'outputs'\n"
            "fit_summary = pd.read_csv(OUTPUTS / 'fit_summary.csv')\n"
            "random_summary = pd.read_csv(OUTPUTS / 'random_search_summary.csv')\n"
            "fit_summary.head()"
        ),
        _markdown_cell("## Best model per fixed scenario by BIC"),
        _code_cell(
            "best_fixed = fit_summary.loc[\n"
            "    fit_summary.groupby('scenario')['bic'].idxmin(),\n"
            "    ['scenario', 'model', 'bic', 'mode_count_grid'],\n"
            "]\n"
            "best_fixed.sort_values('scenario')"
        ),
        _markdown_cell("## Randomized quality-search: best fitted result"),
        _code_cell(
            "best_random = random_summary.sort_values('quality_score', ascending=False).iloc[0]\n"
            "best_random[[\n"
            "    'trial_id', 'true_family', 'model', 'quality_score',\n"
            "    'ise_to_true_pdf', 'bic', 'true_mode_count', 'fitted_mode_count',\n"
            "]]"
        ),
        _markdown_cell("## Top randomized candidates"),
        _code_cell(
            "random_summary.sort_values('quality_score', ascending=False)[[\n"
            "    'trial_id', 'true_family', 'model', 'quality_score',\n"
            "    'ise_to_true_pdf', 'bic', 'true_mode_count', 'fitted_mode_count',\n"
            "]].head(10)"
        ),
        _markdown_cell("## Visual diagnostics"),
        _code_cell(
            "display(Markdown('### Best randomized fit overlay'))\n"
            "display(Image(filename=str(OUTPUTS / 'random_search_best_fit.png')))\n"
            "display(Markdown('### Fixed scenario overlays'))\n"
            "for path in sorted(OUTPUTS.glob('*_fit_overlay.png')):\n"
            "    display(Markdown(f'#### {path.name}'))\n"
            "    display(Image(filename=str(path)))"
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    return notebook


def write_notebook(path: Path = NOTEBOOK_PATH, execute: bool = False) -> Path:
    """Write the notebook, optionally executing it in the project root."""

    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    if execute:
        client = NotebookClient(
            notebook,
            timeout=300,
            kernel_name="python3",
            resources={"metadata": {"path": str(Path.cwd())}},
        )
        client.execute()
    nbformat.write(notebook, path)
    return path


def build_parser() -> argparse.ArgumentParser:
    """Create the notebook-builder argument parser."""

    parser = argparse.ArgumentParser(description="Build the fitting report notebook.")
    parser.add_argument(
        "--execute", action="store_true", help="Execute the notebook before writing."
    )
    parser.add_argument("--output", type=Path, default=NOTEBOOK_PATH, help="Notebook output path.")
    return parser


def main() -> int:
    """Build, and optionally execute, the report notebook."""

    parser = build_parser()
    args = parser.parse_args()
    path = write_notebook(path=args.output, execute=args.execute)
    print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
