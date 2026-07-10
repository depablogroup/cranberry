from __future__ import annotations

import json

import pytest
from openmm import app, unit

from cranberry.data import data_path
from cranberry.remd import (
    RemdRunConfig,
    TemperatureLadderSpec,
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
def test_run_remd_records_online_analysis_interval(tmp_path):
    pdb = data_path('examples/2ntCG_cg_vs_conect.pdb')
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
    args = json.loads(result.args_path.read_text())
    assert args['n_analysis'] == 2
    assert args['online_analysis_interval'] == 1


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
