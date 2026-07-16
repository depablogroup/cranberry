from __future__ import annotations

import os
from collections.abc import Iterable, Mapping


OPENMM_RUNTIME_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_MPS_PIPE_DIRECTORY",
    "CUDA_MPS_LOG_DIRECTORY",
    "PARENT_CUDA_VISIBLE_DEVICES",
)


def normalize_platform_properties(platform_properties: Mapping[str, object] | None) -> dict[str, str] | None:
    """Return OpenMM platform properties as a string dictionary."""

    if not platform_properties:
        return None
    return {str(key): str(value) for key, value in platform_properties.items()}


def parse_platform_property(value: str) -> tuple[str, str]:
    """Parse a KEY=VALUE OpenMM platform property CLI token."""

    key, separator, property_value = value.partition("=")
    if separator != "=" or not key:
        raise ValueError("OpenMM platform properties must use KEY=VALUE")
    return key, property_value


def platform_properties_from_pairs(pairs: Iterable[tuple[str, str]] | None) -> dict[str, str] | None:
    """Build platform properties from parsed CLI pairs; later duplicates win."""

    properties = {key: value for key, value in pairs or ()}
    return properties or None


def openmm_runtime_env_metadata() -> dict[str, str | None]:
    """Capture GPU/MPS environment values that affect OpenMM placement."""

    return {key.lower(): os.environ.get(key) for key in OPENMM_RUNTIME_ENV_KEYS}


def jax_runtime_metadata(*, collect_backend: bool) -> dict[str, object]:
    """Capture JAX metadata without making JAX a hard dependency."""

    metadata: dict[str, object] = {
        "jax_platform_name_env": os.environ.get("JAX_PLATFORM_NAME"),
        "jax_metadata_collected": bool(collect_backend),
        "jax_version": None,
        "jaxlib_version": None,
        "jax_default_backend": None,
        "jax_devices": None,
        "jax_metadata_error": None,
    }
    if not collect_backend:
        return metadata

    try:
        import jax
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        metadata["jax_metadata_error"] = f"{type(exc).__name__}: {exc}"
        return metadata

    metadata["jax_version"] = getattr(jax, "__version__", None)
    try:
        import jaxlib

        metadata["jaxlib_version"] = getattr(jaxlib, "__version__", None)
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        metadata["jax_metadata_error"] = f"{type(exc).__name__}: {exc}"

    try:
        metadata["jax_default_backend"] = jax.default_backend()
        metadata["jax_devices"] = [str(device) for device in jax.devices()]
    except Exception as exc:  # pragma: no cover - backend initialization is host-specific
        metadata["jax_metadata_error"] = f"{type(exc).__name__}: {exc}"
    return metadata
