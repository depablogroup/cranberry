"""CRANBERRY coarse-grained RNA simulation tools."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cranberry-rna")
except PackageNotFoundError:  # pragma: no cover - editable source tree before install
    __version__ = "0+unknown"

from cranberry.forcefield import CranberryForceField

__all__ = ["__version__", "CranberryForceField"]
