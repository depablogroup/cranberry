from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from openmm import app, unit

from cranberry.data import data_path
from cranberry.remd import (
    RemdRunConfig,
    TemperatureLadderSpec,
    _warn_cuda_online_analysis,
    run_remd,
    translate_netcdf_to_dcd,
    _remd_step_plan,
    _run_remd_steps,
)


def _mpirun_or_skip() -> str:
    mpirun = shutil.which('mpirun')
    if mpirun is None:
        pytest.skip('mpirun is required for MPI REMD regression tests')
    return mpirun


def _mpi_env() -> dict[str, str]:
    env = os.environ.copy()
    env['OPENMMTOOLS_ENABLE_MPI'] = '1'
    env.setdefault('JAX_PLATFORM_NAME', 'cpu')
    return env


@pytest.mark.remd
def test_temperature_ladder_spec_default_uses_openmmtools_schedule():
    spec = TemperatureLadderSpec()
    assert spec.resolve() is None
    assert spec.t_min == 298.0
    assert spec.t_max == 600.0
    assert spec.n_replicas == 8


@pytest.mark.remd
def test_temperature_ladder_spec_override_wins():
    spec = TemperatureLadderSpec(temperatures=(300.0, 325.0, 350.0))
    assert spec.resolve() == (300.0, 325.0, 350.0)


def test_remd_step_plan_preserves_requested_total_steps():
    assert _remd_step_plan(1, 5000) == (0, 1, 1)
    assert _remd_step_plan(5001, 5000) == (1, 1, 2)
    assert _remd_step_plan(10000, 5000) == (2, 0, 2)


class _FakeMove:
    def __init__(self):
        self.n_steps = 0


class _FakeSampler:
    def __init__(self, *, iteration=0, moves=None):
        self._iteration = iteration
        self.mcmc_moves = moves or [_FakeMove(), _FakeMove()]
        for move in self.mcmc_moves:
            move.n_steps = 5000
        self.calls = []

    def run(self, n_iterations=None):
        self.calls.append(("run", n_iterations, self.mcmc_moves[0].n_steps))
        self._iteration += n_iterations or 0

    def extend(self, n_iterations):
        self.calls.append(("extend", n_iterations, self.mcmc_moves[0].n_steps))
        self._iteration += n_iterations


def test_run_remd_steps_preserves_non_divisible_step_count():
    sampler = _FakeSampler()

    _run_remd_steps(
        sampler,
        full_iterations=2,
        remainder_steps=3,
        restart=False,
    )

    assert sampler.calls == [("run", 2, 5000), ("extend", 1, 3)]
    assert all(move.n_steps == 3 for move in sampler.mcmc_moves)


def test_run_remd_steps_preserves_short_run_step_count():
    sampler = _FakeSampler()

    _run_remd_steps(
        sampler,
        full_iterations=0,
        remainder_steps=3,
        restart=False,
    )

    assert sampler.calls == [("run", 1, 3)]


@pytest.mark.remd
def test_run_remd_and_translate_real_openmmtools(tmp_path):
    pdb = data_path('examples/2ntCG_cg_vs_conect.pdb')
    result = run_remd(
        RemdRunConfig(
            pdb_path=pdb,
            steps=1,
            output_dir=tmp_path,
            temperature_ladder=TemperatureLadderSpec(temperatures=(298.0, 318.0)),
            swap_steps=1,
            overwrite=True,
        )
    )
    assert result.output_netcdf_path.exists()
    assert result.output_netcdf_path.stat().st_size > 0
    assert result.args_path.exists()
    args = json.loads(result.args_path.read_text())
    assert args['run_kind'] == 'remd'
    assert args['steps'] == 1
    assert args['iterations'] == 1
    assert args['swap_steps'] == 1
    assert args['checkpoint_interval'] == 1
    assert args['online_analysis_interval'] is None
    assert args['temperature_ladder_kelvin'] == [298.0, 318.0]
    assert args['platform'] == 'CPU'
    assert args['actual_platform'] == 'CPU'
    assert args['periodic'] is False
    assert args['periodic_box_vectors_present'] is False
    assert args['mcmc_move'] == 'LangevinDynamicsMove'
    assert len(args['langevin_collision_rate_per_ps']) == 2
    assert all(rate > 0 for rate in args['langevin_collision_rate_per_ps'])

    output_dcds = translate_netcdf_to_dcd(
        result.output_netcdf_path,
        pdb_path=pdb,
        output_dir=tmp_path,
        output_mode='replica',
        overwrite=True,
    )
    assert isinstance(output_dcds, tuple)
    assert (tmp_path / 'output_0.dcd').exists()
    assert (tmp_path / 'output_1.dcd').exists()


@pytest.mark.remd
def test_translate_netcdf_to_dcd_temperature_mode(tmp_path):
    pdb = data_path('examples/2ntCG_cg_vs_conect.pdb')
    result = run_remd(
        RemdRunConfig(
            pdb_path=pdb,
            steps=1,
            output_dir=tmp_path,
            temperature_ladder=TemperatureLadderSpec(temperatures=(298.0, 318.0)),
            swap_steps=1,
            overwrite=True,
        )
    )
    output_dcds = translate_netcdf_to_dcd(
        result.output_netcdf_path,
        pdb_path=pdb,
        output_dir=tmp_path,
        output_mode='temperature',
        overwrite=True,
    )
    assert output_dcds == (tmp_path / 'output_T0.dcd', tmp_path / 'output_T1.dcd')
    assert (tmp_path / 'output_T0.dcd').exists()
    assert (tmp_path / 'output_T1.dcd').exists()
    manifest = tmp_path / 'output_temperature_labels.txt'
    assert manifest.exists()
    assert 'T0 = 298.000 K' in manifest.read_text()


@pytest.mark.remd
def test_run_remd_periodic_extra_start_uses_one_common_box(tmp_path):
    pdb = data_path('examples/2ntCG_cg_vs_conect.pdb')
    source = app.PDBFile(str(pdb))
    coordinates = source.positions.value_in_unit(unit.nanometer)
    center = np.mean(coordinates, axis=0)
    expanded_positions = (center + 2.0 * (coordinates - center)) * unit.nanometer
    extra = tmp_path / 'expanded.pdb'
    with extra.open('w') as handle:
        app.PDBFile.writeFile(source.topology, expanded_positions, handle, keepIds=True)
    conect = ''.join(
        line
        for line in Path(pdb).read_text().splitlines(keepends=True)
        if line.startswith('CONECT')
    )
    extra.write_text(extra.read_text().replace('END\n', conect + 'END\n'))

    expanded = app.PDBFile(str(extra))
    expanded_coordinates = expanded.positions.value_in_unit(unit.nanometer)
    expanded_span = float(
        np.max(np.max(expanded_coordinates, axis=0) - np.min(expanded_coordinates, axis=0))
    )
    expected_box_size = expanded_span + 5.0

    result = run_remd(
        RemdRunConfig(
            pdb_path=pdb,
            steps=1,
            output_dir=tmp_path,
            temperature_ladder=TemperatureLadderSpec(temperatures=(298.0, 318.0)),
            swap_steps=1,
            extra_start_pdb=extra,
            periodic=True,
            box_padding_nanometer=2.5,
            overwrite=True,
        )
    )

    args = json.loads(result.args_path.read_text())
    vectors = np.asarray(args['periodic_box_vectors_nanometer'])
    assert vectors == pytest.approx(np.diag([expected_box_size] * 3))
    assert args['periodic_box_vectors_present'] is True


def test_run_remd_rejects_extra_start_with_different_topology(tmp_path):
    with pytest.raises(ValueError, match='same ordered atoms and bonds'):
        run_remd(
            RemdRunConfig(
                pdb_path=data_path('examples/2ntCG_cg_vs_conect.pdb'),
                extra_start_pdb=data_path('examples/1zih_cg_vs_conect.pdb'),
                steps=1,
                output_dir=tmp_path,
                overwrite=True,
            )
        )


@pytest.mark.remd
def test_run_remd_accepts_extra_start_pdb_and_records_metadata(tmp_path):
    pdb = data_path('examples/2ntCG_cg_vs_conect.pdb')
    result = run_remd(
        RemdRunConfig(
            pdb_path=pdb,
            steps=1,
            output_dir=tmp_path,
            temperature_ladder=TemperatureLadderSpec(temperatures=(298.0, 318.0)),
            swap_steps=1,
            extra_start_pdb=pdb,
            overwrite=True,
        )
    )
    args = json.loads(result.args_path.read_text())
    assert args['extra_start_pdb_path'] == str(pdb)
    assert args['extra_start_pdb_sha256']


def test_warn_cuda_online_analysis_reports_jax_gpu_risk():
    with pytest.warns(RuntimeWarning, match='PyMBAR/JAX.*CUDA.*--n-analysis 0'):
        _warn_cuda_online_analysis('CUDA', 1)


def test_warn_cuda_online_analysis_ignores_disabled_analysis():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        _warn_cuda_online_analysis('CUDA', None)
    assert not captured


@pytest.mark.remd
def test_run_remd_records_online_analysis_interval(tmp_path, monkeypatch):
    pdb = data_path('examples/2ntCG_cg_vs_conect.pdb')
    monkeypatch.setenv('JAX_PLATFORM_NAME', 'cpu')
    result = run_remd(
        RemdRunConfig(
            pdb_path=pdb,
            steps=2,
            output_dir=tmp_path,
            temperature_ladder=TemperatureLadderSpec(temperatures=(298.0, 318.0)),
            swap_steps=1,
            n_analysis=2,
            overwrite=True,
        )
    )
    assert result.online_analysis_interval == 1
    assert result.jax_platform_name_env == 'cpu'
    args = json.loads(result.args_path.read_text())
    assert args['n_analysis'] == 2
    assert args['online_analysis_interval'] == 1
    assert args['jax_platform_name_env'] == 'cpu'


@pytest.mark.remd
def test_run_remd_rejects_ladder_mismatch_on_restart(tmp_path):
    pdb = data_path('examples/2ntCG_cg_vs_conect.pdb')
    first = run_remd(
        RemdRunConfig(
            pdb_path=pdb,
            steps=1,
            output_dir=tmp_path,
            temperature_ladder=TemperatureLadderSpec(temperatures=(298.0, 318.0)),
            swap_steps=1,
            overwrite=True,
        )
    )

    with pytest.raises(ValueError, match='ladder does not match'):
        run_remd(
            RemdRunConfig(
                pdb_path=pdb,
                steps=1,
                output_dir=tmp_path,
                temperature_ladder=TemperatureLadderSpec(temperatures=(298.0, 330.0)),
                swap_steps=1,
                restart_from=first.output_netcdf_path,
                overwrite=True,
            )
        )


@pytest.mark.remd
def test_run_remd_rejects_incompatible_restart_metadata(tmp_path):
    pdb = data_path('examples/2ntCG_cg_vs_conect.pdb')
    first = run_remd(
        RemdRunConfig(
            pdb_path=pdb,
            steps=1,
            output_dir=tmp_path,
            temperature_ladder=TemperatureLadderSpec(temperatures=(298.0, 318.0)),
            swap_steps=1,
            overwrite=True,
        )
    )

    with pytest.raises(ValueError, match='restart configuration.*periodic'):
        run_remd(
            RemdRunConfig(
                pdb_path=pdb,
                steps=1,
                temperature_ladder=TemperatureLadderSpec(temperatures=(298.0, 318.0)),
                swap_steps=1,
                restart_from=first.output_netcdf_path,
                periodic=True,
                overwrite=True,
            )
        )


@pytest.mark.remd
def test_run_remd_mpi_restart_reopens_existing_storage(tmp_path):
    """Reproduce the MPI NetCDF restart-open failure tracked in issue #9."""

    mpirun = _mpirun_or_skip()

    pdb = data_path('examples/2ntCG_cg_vs_conect.pdb')
    first = run_remd(
        RemdRunConfig(
            pdb_path=pdb,
            steps=1,
            output_dir=tmp_path,
            temperature_ladder=TemperatureLadderSpec(temperatures=(298.0, 318.0)),
            swap_steps=1,
            overwrite=True,
        )
    )

    command = [
        mpirun,
        '--oversubscribe',
        '-n',
        '2',
        sys.executable,
        '-m',
        'cranberry.cli.main',
        'remd',
        str(pdb),
        '--steps',
        '1',
        '--swap-steps',
        '1',
        '--temperature-ladder',
        '298',
        '318',
        '--restart-from',
        str(first.output_netcdf_path),
        '--platform',
        'CPU',
        '--overwrite',
    ]

    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=_mpi_env(),
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, (
        'MPI REMD restart failed while reopening existing OpenMMTools NetCDF storage.\n'
        f'command: {command}\n'
        f'stdout:\n{completed.stdout}\n'
        f'stderr:\n{completed.stderr}'
    )


@pytest.mark.remd
def test_run_remd_mpi_fresh_run_postprocess_is_rank_safe(tmp_path):
    """Verify fresh MPI REMD post-processing is rank-safe."""

    mpirun = _mpirun_or_skip()
    pdb = data_path('examples/2ntCG_cg_vs_conect.pdb')
    command = [
        mpirun,
        '--oversubscribe',
        '-n',
        '2',
        sys.executable,
        '-m',
        'cranberry.cli.main',
        'remd',
        str(pdb),
        '--steps',
        '1',
        '--swap-steps',
        '1',
        '--temperature-ladder',
        '298',
        '318',
        '--output-dir',
        str(tmp_path),
        '--platform',
        'CPU',
        '--overwrite',
    ]

    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=_mpi_env(),
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, (
        'Fresh MPI REMD should finish post-run metadata collection on all ranks.\n'
        f'command: {command}\n'
        f'stdout:\n{completed.stdout}\n'
        f'stderr:\n{completed.stderr}'
    )


@pytest.mark.remd
def test_remd_extract_mpi_runs_only_on_root_rank(tmp_path):
    """Verify MPI-launched DCD extraction runs only on the root rank."""

    mpirun = _mpirun_or_skip()
    pdb = data_path('examples/2ntCG_cg_vs_conect.pdb')
    result = run_remd(
        RemdRunConfig(
            pdb_path=pdb,
            steps=1,
            output_dir=tmp_path / 'source',
            temperature_ladder=TemperatureLadderSpec(temperatures=(298.0, 318.0)),
            swap_steps=1,
            overwrite=True,
        )
    )
    output_dir = tmp_path / 'extract'
    command = [
        mpirun,
        '--oversubscribe',
        '-n',
        '2',
        sys.executable,
        '-m',
        'cranberry.cli.main',
        'remd-extract',
        str(result.output_netcdf_path),
        str(pdb),
        '--output-dir',
        str(output_dir),
        '--overwrite',
    ]

    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=_mpi_env(),
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, (
        'MPI remd-extract should not fail when launched under mpirun.\n'
        f'command: {command}\n'
        f'stdout:\n{completed.stdout}\n'
        f'stderr:\n{completed.stderr}'
    )
    assert completed.stdout.count('remd-extract mode: replica') == 1
