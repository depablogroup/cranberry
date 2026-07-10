"""CRANBERRY coarse-grained RNA simulation tools."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cranberry-rna")
except PackageNotFoundError:  # pragma: no cover - editable source tree before install
    __version__ = "0+unknown"

from cranberry.cg import CoarseGrainResult, coarse_grain_structure
from cranberry.forcefield import CranberryForceField
from cranberry.md import DetailedEnergyReporter, MDRunResult, calculate_langevin_friction, create_simulation, run_md
from cranberry.prepare import PreparedStructureResult, prepare_structure
from cranberry.remd import RemdRunConfig, RemdRunResult, TemperatureLadderSpec, run_remd, translate_netcdf_to_dcd

__all__ = [
    "__version__",
    "CoarseGrainResult",
    "CranberryForceField",
    "DetailedEnergyReporter",
    "MDRunResult",
    "PreparedStructureResult",
    "RemdRunConfig",
    "RemdRunResult",
    "TemperatureLadderSpec",
    "calculate_langevin_friction",
    "coarse_grain_structure",
    "create_simulation",
    "prepare_structure",
    "run_md",
    "run_remd",
    "translate_netcdf_to_dcd",
]
