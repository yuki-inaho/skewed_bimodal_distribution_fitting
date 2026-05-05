"""Distribution model registry: name → spec lookup for the fitting pipeline.

(layer: kernel) The registry decouples *which* models exist from *how* their
densities are computed. The numerical kernels live in
:mod:`bimodal_skewfit.distributions`; this module only owns the mapping
``name -> (param_names, logpdf, k)``. In a future Rust port, this is the table
that becomes a ``Distribution`` enum with one variant per family, each
carrying its parameter struct and dispatching to the corresponding kernel
function.
"""

from __future__ import annotations

from dataclasses import dataclass

from .distributions import (
    LogPdf,
    abn_logpdf,
    adn_logpdf,
    bsn_fs_logpdf,
    gmm2_logpdf,
    normal_logpdf,
    ntpn_logpdf,
)


@dataclass(frozen=True, slots=True)
class DistributionSpec:
    """Metadata required to fit and report one distribution family."""

    name: str
    param_names: tuple[str, ...]
    logpdf: LogPdf
    num_parameters: int

    def __post_init__(self) -> None:
        if self.num_parameters != len(self.param_names):
            raise ValueError(
                f"DistributionSpec({self.name!r}): num_parameters={self.num_parameters} "
                f"does not match len(param_names)={len(self.param_names)}"
            )


DISTRIBUTIONS: dict[str, DistributionSpec] = {
    "normal": DistributionSpec("normal", ("loc", "scale"), normal_logpdf, 2),
    "gmm2": DistributionSpec(
        "gmm2",
        ("weight", "mu1", "sigma1", "mu2", "sigma2"),
        gmm2_logpdf,
        5,
    ),
    "abn": DistributionSpec("abn", ("loc", "scale", "lambda", "alpha"), abn_logpdf, 4),
    "adn": DistributionSpec("adn", ("loc", "scale", "sep", "skew"), adn_logpdf, 4),
    "bsn_fs": DistributionSpec(
        "bsn_fs",
        ("loc", "scale", "alpha", "gamma"),
        bsn_fs_logpdf,
        4,
    ),
    "ntpn": DistributionSpec("ntpn", ("loc", "scale", "alpha", "lambda"), ntpn_logpdf, 4),
}
