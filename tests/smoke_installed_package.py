"""Smoke test executed against an installed wheel or sdist.

Run from the project root via the ``just package-smoke`` recipe, which
isolates Python from the source checkout and installs only the built
distribution. The script intentionally uses just the public API and core
dependencies (numpy, scipy) so that it fails loudly if the wheel happens
to drag in pandas/matplotlib by accident.

This file is named ``smoke_*.py`` rather than ``test_*.py`` so that the
default pytest collector ignores it; only the dedicated recipe runs it.
"""

from __future__ import annotations

import numpy as np

import bimodal_skewfit as bsf


def main() -> None:
    rng = np.random.default_rng(20260605)
    sample = rng.normal(size=200)

    assert isinstance(bsf.__version__, str), "bsf.__version__ must be a string"
    models = bsf.list_models()
    assert "normal" in models, f"normal missing from list_models(): {models}"
    assert "gmm2" in models, f"gmm2 missing from list_models(): {models}"

    result = bsf.fit_model("normal", sample)
    assert result.model == "normal"
    assert np.isfinite(result.nll)
    assert np.isfinite(result.bic)

    print(f"OK: bimodal-skewfit=={bsf.__version__}, models={models}, normal NLL={result.nll:.4f}")


if __name__ == "__main__":
    main()
