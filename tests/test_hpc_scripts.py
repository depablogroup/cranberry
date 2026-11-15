from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from cranberry.data import data_path


REPO_ROOT = Path(__file__).resolve().parents[1]

CONCURRENT_MD_SCRIPT = REPO_ROOT / "scripts/hpc/submit_md_mps_concurrent_example_slurm.sh"


def test_md_mps_concurrent_example_slurm_parses_with_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(CONCURRENT_MD_SCRIPT)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_md_mps_concurrent_example_slurm_dry_run(tmp_path: Path) -> None:
    pdb_a = data_path("examples/2ntCG_cg_vs_conect.pdb")
    pdb_b = data_path("examples/157d_cg_vs_conect.pdb")
    output_root = tmp_path / "md_concurrent"

    result = subprocess.run(
        [
            str(CONCURRENT_MD_SCRIPT),
            "--dry-run",
            "--pdb",
            str(pdb_a),
            "--pdb",
            str(pdb_b),
            "--run-name",
            "tiny_duplex",
            "--run-name",
            "tiny_single",
            "--output-root",
            str(output_root),
            "--account",
            "example_account",
            "--partition",
            "example_gpu",
            "--steps",
            "100",
            "--n-record",
            "10",
        ],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "pdb_count=2" in result.stdout
    assert "dry-run sbatch:" in result.stdout
    assert "--account=example_account" in result.stdout
    assert "--partition=example_gpu" in result.stdout
    assert "--gres=gpu:1" in result.stdout
    assert f"--chdir={output_root}" in result.stdout
    assert str(CONCURRENT_MD_SCRIPT) in result.stdout


def test_md_mps_concurrent_example_slurm_runs_md_directly() -> None:
    script_text = CONCURRENT_MD_SCRIPT.read_text()

    assert '"$CRANBERRY_BIN" md "$pdb_path"' in script_text
    assert "CRANBERRY_MD_HELPER" not in script_text


REMD_MPS_SCRIPT = REPO_ROOT / "scripts/hpc/submit_remd_mpi_mps_one_gpu_example_slurm.sh"


def test_remd_mpi_mps_one_gpu_example_parses_with_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(REMD_MPS_SCRIPT)], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert "#SBATCH --gres=gpu:1" in REMD_MPS_SCRIPT.read_text()
    assert "#SBATCH --ntasks=8" in REMD_MPS_SCRIPT.read_text()


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("launch_status", [0, 7])
def test_remd_mpi_mps_launch_and_cleanup(
    tmp_path: Path, override: bool, launch_status: int
) -> None:
    import json

    # Run the batch body with fake tools: no GPU, MPI, or scheduler required.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    events = tmp_path / "events"
    launcher = bin_dir / "srun"
    launcher.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['LAUNCH_RECORD']).write_text(json.dumps({"
        "'argv': sys.argv[1:], 'mpi': os.environ['OPENMMTOOLS_ENABLE_MPI'], "
        "'pipe': os.environ['CUDA_MPS_PIPE_DIRECTORY'], "
        "'log': os.environ['CUDA_MPS_LOG_DIRECTORY']}))\n"
        "sys.exit(int(os.environ['LAUNCH_STATUS']))\n"
    )
    launcher.chmod(0o755)
    mps = bin_dir / "nvidia-cuda-mps-control"
    mps.write_text(
        '#!/usr/bin/env bash\n'
        'if [ "${1:-}" = "-d" ]; then\n'
        '    echo start >> "$EVENTS"\n'
        'else\n'
        '    cat >> "$EVENTS"\n'
        'fi\n'
    )
    mps.chmod(0o755)
    env = os.environ.copy()
    for key in (
        "CRANBERRY_BIN", "PDB_PATH", "EXTRA_START_PDB", "OUTPUT_DIR", "STEPS",
        "N_REPLICAS", "SLURM_NTASKS", "OPENMMTOOLS_ENABLE_MPI",
        "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY",
    ):
        env.pop(key, None)
    pipe_dir = tmp_path / "mps pipe"
    log_dir = tmp_path / "mps log"
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}", EVENTS=str(events),
        LAUNCH_RECORD=str(tmp_path / "launch.json"), LAUNCH_STATUS=str(launch_status),
        CUDA_MPS_PIPE_DIRECTORY=str(pipe_dir), CUDA_MPS_LOG_DIRECTORY=str(log_dir),
    )
    replicas, steps = ("4", "500000") if override else ("8", "100000")
    pdb = "input with spaces.pdb" if override else "cranberry/data/examples/ggcGCAAgcc_cg_vs_conect.pdb"
    extra = "extra with spaces.pdb" if override else "cranberry/data/examples/ggcGCAAgcc_extended_cg_vs_conect.pdb"
    output = "custom output" if override else "remd-run"
    executable = "custom cranberry" if override else "cranberry"
    if override:
        env.update(
            N_REPLICAS=replicas, SLURM_NTASKS=replicas, STEPS=steps,
            PDB_PATH=pdb, EXTRA_START_PDB=extra, OUTPUT_DIR=output,
            CRANBERRY_BIN=executable,
        )
    result = subprocess.run(
        ["bash", str(REMD_MPS_SCRIPT)], cwd=tmp_path, env=env,
        text=True, capture_output=True,
    )
    assert result.returncode == launch_status, result.stderr
    record = json.loads((tmp_path / "launch.json").read_text())
    assert record == {
        "argv": ["-n", replicas, executable, "remd", pdb, "--extra-start-pdb", extra,
                 "--periodic", "--steps", steps, "--n-replicas", replicas,
                 "--platform", "CUDA", "--output-dir", output],
        "mpi": "1", "pipe": str(pipe_dir), "log": str(log_dir),
    }
    assert (tmp_path / output).is_dir()
    assert pipe_dir.is_dir() and log_dir.is_dir()
    assert events.read_text().splitlines() == ["start", "quit"]
