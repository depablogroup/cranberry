import pytest
import openmm as mm

from cranberry.data import data_path
from cranberry.md import create_simulation


class _UnsupportedPtxContext:
    def setPositions(self, positions):
        raise mm.OpenMMException(
            "Error compiling CUDA kernel: CUDA_ERROR_UNSUPPORTED_PTX_VERSION "
            "(the provided PTX was compiled with an unsupported toolchain)"
        )

    def setVelocitiesToTemperature(self, temperature):  # pragma: no cover - error occurs first
        raise AssertionError("setVelocitiesToTemperature should not run after the PTX failure")


class _UnsupportedPtxSimulation:
    def __init__(self, *args, **kwargs):
        self.context = _UnsupportedPtxContext()


def test_cuda_unsupported_ptx_version_surfaces_from_context_creation(monkeypatch):
    """Reproduce the CUDA 13/NVRTC 13 unsupported-PTX failure as OpenMM raises it.

    The test intentionally documents the current buggy behavior: Cranberry does not
    yet translate this CUDA/OpenMM exception into a Cranberry-specific diagnostic.
    """

    monkeypatch.setattr("cranberry.md.app.Simulation", _UnsupportedPtxSimulation)
    monkeypatch.setattr("cranberry.md.Platform.getPlatformByName", lambda name: object())

    with pytest.raises(mm.OpenMMException, match="CUDA_ERROR_UNSUPPORTED_PTX_VERSION"):
        create_simulation(data_path("examples/2ntCG_cg_vs_conect.pdb"), platform="CUDA")
