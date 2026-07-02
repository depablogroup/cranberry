from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cranberry.data import data_path

FORCE_GROUP_NAMES = (
    "bond",
    "angle",
    "dihedral",
    "pucker",
    "stacking35",
    "stacking55",
    "stacking33",
    "pairing",
    "wca",
    "spline",
    "electrostatic",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    parameter_file: str
    xml_file: str
    description: str

    @property
    def parameter_path(self) -> Path:
        return data_path(self.parameter_file)

    @property
    def xml_path(self) -> Path:
        return data_path(self.xml_file)


_MODEL_REGISTRY = {
    "cranberry-v1-alpha.1": ModelSpec(
        name="cranberry-v1-alpha.1",
        parameter_file="forcefields/cranberry-v1-alpha.1.h5",
        xml_file="xml/cranberry.xml",
        description="CRANBERRY v1 alpha model bundle.",
    ),
}
_DEFAULT_MODEL = "cranberry-v1-alpha.1"


def available_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def default_model_name() -> str:
    return _DEFAULT_MODEL


def get_model_spec(name: str = "default") -> ModelSpec:
    if name == "default":
        name = _DEFAULT_MODEL
    try:
        return _MODEL_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(["default", *available_models()])
        raise ValueError(f"Unknown CRANBERRY model {name!r}. Available models: {known}") from exc
