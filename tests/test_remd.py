from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from openmm import app, unit

from cranberry.data import data_path
from cranberry.remd import (
    RemdRunConfig,
    TemperatureLadderSpec,
    _warn_cuda_online_analysis,
    run_remd,
    translate_netcdf_to_dcd,
)


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
def test_run_remd_mpi_restart_reopens_existing_storage(tmp_path):
    """Reproduce the MPI NetCDF restart-open failure tracked in issue #9."""

    mpirun = shutil.which('mpirun')
    if mpirun is None:
        pytest.skip('mpirun is required for the MPI REMD restart regression')

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

    env = os.environ.copy()
    env['OPENMMTOOLS_ENABLE_MPI'] = '1'
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
        env=env,
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
