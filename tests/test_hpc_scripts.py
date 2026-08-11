from __future__ import annotations

import os
import subprocess
from pathlib import Path

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
