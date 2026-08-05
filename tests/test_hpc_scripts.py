from __future__ import annotations

from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
REMD_MPS_SCRIPT = REPO_ROOT / "scripts/hpc/submit_remd_mpi_mps_one_gpu_example_slurm.sh"


def test_remd_mpi_mps_one_gpu_example_parses_with_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(REMD_MPS_SCRIPT)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_remd_mpi_mps_one_gpu_example_requests_one_gpu() -> None:
    script_text = REMD_MPS_SCRIPT.read_text()

    assert "#SBATCH --gres=gpu:1" in script_text
    assert "#SBATCH --ntasks=8" in script_text


def test_remd_mpi_mps_one_gpu_example_runs_remd_directly() -> None:
    script_text = REMD_MPS_SCRIPT.read_text()

    assert 'srun -n "$N_REPLICAS" "$CRANBERRY_BIN" remd "$PDB_PATH"' in script_text
    assert "OPENMMTOOLS_ENABLE_MPI" in script_text
    assert "nvidia-cuda-mps-control -d" in script_text
    assert "--extra-start-pdb" in script_text
    assert "--periodic" in script_text
