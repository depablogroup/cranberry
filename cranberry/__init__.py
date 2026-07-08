"""CRANBERRY coarse-grained RNA simulation tools."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cranberry-rna")
except PackageNotFoundError:  # pragma: no cover - editable source tree before install
    __version__ = "0+unknown"

from cranberry.forcefield import CranberryForceField
from cranberry.md import DetailedEnergyReporter, MDRunResult, calculate_langevin_friction, create_simulation, run_md
from cranberry.prepare import PreparedStructureResult, prepare_structure

__all__ = [
    "__version__",
    "CranberryForceField",
    "DetailedEnergyReporter",
    "MDRunResult",
    "calculate_langevin_friction",
    "create_simulation",
    "PreparedStructureResult",
    "prepare_structure",
    "run_md",
]
