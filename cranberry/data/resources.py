from __future__ import annotations

from contextlib import ExitStack
from importlib import resources
from pathlib import Path

_DATA_PACKAGE = "cranberry.data"
_STACK = ExitStack()


def data_path(relative_path: str) -> Path:
    """Return a filesystem path for a packaged data file.

    The returned path is suitable for libraries that require a real path rather
    than a package resource object. Extracted temporary resources are kept alive
    for the process lifetime.
    """

    resource = resources.files(_DATA_PACKAGE).joinpath(relative_path)
    if not resource.is_file():
        raise FileNotFoundError(f"Packaged data file not found: {relative_path}")
    return Path(_STACK.enter_context(resources.as_file(resource)))


def available_forcefields() -> list[str]:
    root = resources.files(_DATA_PACKAGE).joinpath("forcefields")
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.name.endswith(".h5"))
