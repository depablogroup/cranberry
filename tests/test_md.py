import json

import pytest
from openmm import app, unit

from cranberry.data import data_path
from cranberry.forcefield import CranberryForceField
from cranberry.md import calculate_langevin_friction, create_simulation, run_md


def test_langevin_friction_matches_legacy_formula():
    pdb = app.PDBFile(str(data_path("examples/2ntCG_cg_vs_conect.pdb")))
    system = CranberryForceField().createSystem(pdb.topology, positions=pdb.positions)
    gamma = calculate_langevin_friction(pdb.topology, system, 298 * unit.kelvin)

    diffusion = 4.58e-10 * unit.meter**2 / unit.second * 2 ** (-0.39)
    total_mass = sum((system.getParticleMass(i) for i in range(system.getNumParticles())), 0 * unit.dalton)
    total_mass = total_mass.in_units_of(unit.gram / unit.mole)
    expected = (unit.MOLAR_GAS_CONSTANT_R * 298 * unit.kelvin / (diffusion * total_mass)).in_units_of(unit.picosecond**-1)

    assert gamma.value_in_unit(unit.picosecond**-1) == pytest.approx(expected.value_in_unit(unit.picosecond**-1))


def test_create_simulation_uses_requested_temperature_and_timestep():
    simulation = create_simulation(
        data_path("examples/2ntCG_cg_vs_conect.pdb"),
        temperature=300 * unit.kelvin,
        timestep=5 * unit.femtosecond,
        platform="CPU",
    )
    integrator = simulation.integrator
    assert integrator.getTemperature().value_in_unit(unit.kelvin) == pytest.approx(300)
    assert integrator.getStepSize().value_in_unit(unit.femtosecond) == pytest.approx(5)


def test_run_md_writes_default_outputs(tmp_path):
    result = run_md(
        data_path("examples/2ntCG_cg_vs_conect.pdb"),
        steps=1,
        report_interval=1,
        output_dir=tmp_path,
        platform="CPU",
    )

    for path in [
        result.dcd_path,
        result.log_path,
        result.detailed_log_path,
        result.args_path,
        result.checkpoint_path,
        result.final_pdb_path,
    ]:
        assert path.exists()
        assert path.stat().st_size > 0

    args = json.loads(result.args_path.read_text())
    assert args["steps"] == 1
    assert args["report_interval"] == 1
    assert args["model"] == "cranberry-v1-alpha.1"

    detailed = result.detailed_log_path.read_text().splitlines()
    assert detailed[0].startswith('#"Step","Time (ps)","Potential Energy (kJ/mole)","bond (kJ/mole)"')
    assert len(detailed) == 2


def test_run_md_no_overwrite_rejects_existing_outputs(tmp_path):
    run_md(data_path("examples/2ntCG_cg_vs_conect.pdb"), steps=1, report_interval=1, output_dir=tmp_path, platform="CPU")
    with pytest.raises(FileExistsError):
        run_md(
            data_path("examples/2ntCG_cg_vs_conect.pdb"),
            steps=1,
            report_interval=1,
            output_dir=tmp_path,
            platform="CPU",
            overwrite=False,
        )
