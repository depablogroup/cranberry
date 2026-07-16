from __future__ import annotations

import json
import warnings

import pytest
from openmm import app, unit

from cranberry.data import data_path
from cranberry.remd import (
    RemdRunConfig,
    TemperatureLadderSpec,
    _MpiRuntime,
    _detect_mpi_runtime,
    _mpi_run_single_node,
    _sampler_from_storage,
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
    assert args['platform_properties'] is None
    assert args['actual_platform'] == 'CPU'
    assert args['mpi']['enabled'] is False
    assert args['mpi']['rank'] == 0
    assert args['mpi']['size'] == 1
    assert args['jax_metadata_collected'] is False
    assert 'jax_version' in args
    assert 'cuda_visible_devices' in args
    assert 'cuda_mps_pipe_directory' in args
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


@pytest.mark.remd
def test_run_remd_records_platform_properties(tmp_path):
    pdb = data_path('examples/2ntCG_cg_vs_conect.pdb')
    result = run_remd(
        RemdRunConfig(
            pdb_path=pdb,
            steps=1,
            output_dir=tmp_path,
            temperature_ladder=TemperatureLadderSpec(temperatures=(298.0, 318.0)),
            swap_steps=1,
            platform_properties={'Threads': '1'},
            overwrite=True,
        )
    )
    assert result.platform_properties == {'Threads': '1'}
    args = json.loads(result.args_path.read_text())
    assert args['platform_properties'] == {'Threads': '1'}


def test_detect_mpi_runtime_from_fake_mpiplus():
    class FakeComm:
        def Get_rank(self):
            return 2

        def Get_size(self):
            return 4

    class FakeMpiplus:
        @staticmethod
        def get_mpicomm():
            return FakeComm()

    runtime = _detect_mpi_runtime(FakeMpiplus())
    assert runtime.enabled is True
    assert runtime.rank == 2
    assert runtime.size == 4


def test_mpi_run_single_node_broadcasts_root_exception():
    class FakeMpiplus:
        @staticmethod
        def run_single_node(rank, task, *args, **kwargs):
            assert rank == 0
            assert kwargs['broadcast_result'] is True
            return task(*args)

    def fail():
        raise FileExistsError('already here')

    with pytest.raises(FileExistsError, match='already here'):
        _mpi_run_single_node(FakeMpiplus(), _MpiRuntime(enabled=True, rank=1, size=2), fail)


def test_sampler_from_storage_barriers_before_append_open():
    events = []

    class FakeReporter:
        def open(self, mode):
            events.append(f'open:{mode}')

        def close(self):
            events.append('close')

    class FakeSampler:
        def _restore_sampler_from_reporter(self, reporter):
            events.append('restore')

    class FakeSamplerClass:
        @staticmethod
        def _reporter_from_storage(storage, check_exist=True):
            events.append(f'reporter:{storage}:{check_exist}')
            return FakeReporter()

        @staticmethod
        def _instantiate_sampler_from_reporter(reporter):
            events.append('instantiate')
            return FakeSampler()

    class FakeComm:
        def barrier(self):
            events.append('barrier')

    class FakeMpiplus:
        @staticmethod
        def get_mpicomm():
            return FakeComm()

        @staticmethod
        def run_single_node(rank, task, *args, **kwargs):
            assert rank == 0
            assert kwargs['sync_nodes'] is True
            events.append('run_single_node')
            task_kwargs = {key: value for key, value in kwargs.items() if key not in {'broadcast_result', 'sync_nodes'}}
            return task(*args, **task_kwargs)

    sampler = _sampler_from_storage(FakeSamplerClass, 'storage.nc', FakeMpiplus(), _MpiRuntime(enabled=True, rank=1, size=2))

    assert sampler._reporter is not None
    assert events == [
        'reporter:storage.nc:True',
        'open:r',
        'instantiate',
        'restore',
        'close',
        'barrier',
        'run_single_node',
        'open:a',
    ]


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
    assert args['jax_metadata_collected'] is True
    assert args['jax_default_backend'] == 'cpu'
    assert isinstance(args['jax_devices'], list)


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
def test_run_remd_restarts_from_existing_storage(tmp_path):
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

    second = run_remd(
        RemdRunConfig(
            pdb_path=pdb,
            steps=1,
            output_dir=tmp_path,
            temperature_ladder=TemperatureLadderSpec(temperatures=(298.0, 318.0)),
            swap_steps=1,
            restart_from=first.output_netcdf_path,
            overwrite=True,
        )
    )

    args = json.loads(second.args_path.read_text())
    assert second.restart_from_path == first.output_netcdf_path
    assert second.output_netcdf_path == first.output_netcdf_path
    assert args['restart_from'] == str(first.output_netcdf_path)
