import csv
import json

import numpy as np
import pytest
from openmm import app, unit

import cranberry.md as md_module
from cranberry.data import data_path
from cranberry.forcefield import (
    CranberryForceField,
    prepare_common_periodic_positions,
    validate_periodic_box_cutoffs,
)
from cranberry.md import calculate_langevin_friction, create_simulation, run_md
from cranberry.validation import validate_canonical_pdb




def _force_by_name(system, name):
    matches = [system.getForce(index) for index in range(system.getNumForces()) if system.getForce(index).getName() == name]
    assert matches, f"missing force {name}"
    return matches[0]


def _forces_by_name(system, name):
    return [system.getForce(index) for index in range(system.getNumForces()) if system.getForce(index).getName() == name]


def test_create_system_periodic_switches_legacy_pbc_forces():
    pdb = app.PDBFile(str(data_path("examples/2ntCG_cg_vs_conect.pdb")))
    nonperiodic = CranberryForceField().createSystem(pdb.topology, positions=pdb.positions)

    periodic_pdb = app.PDBFile(str(data_path("examples/2ntCG_cg_vs_conect.pdb")))
    periodic = CranberryForceField().createSystem(periodic_pdb.topology, positions=periodic_pdb.positions, periodic=True)

    for name in ["wca", "spline", "electrostatic"]:
        assert _force_by_name(nonperiodic, name).getNonbondedMethod() == _force_by_name(nonperiodic, name).CutoffNonPeriodic
        assert _force_by_name(periodic, name).getNonbondedMethod() == _force_by_name(periodic, name).CutoffPeriodic

    for name in ["stacking", "pairing"]:
        assert _force_by_name(nonperiodic, name).getNonbondedMethod() == _force_by_name(nonperiodic, name).CutoffNonPeriodic
        assert _force_by_name(periodic, name).getNonbondedMethod() == _force_by_name(periodic, name).CutoffPeriodic

    assert all(not force.usesPeriodicBoundaryConditions() for force in _forces_by_name(nonperiodic, "pucker"))
    assert all(force.usesPeriodicBoundaryConditions() for force in _forces_by_name(periodic, "pucker"))
    assert not _force_by_name(periodic, "bond").usesPeriodicBoundaryConditions()
    assert not _force_by_name(periodic, "angle").usesPeriodicBoundaryConditions()
    assert not _force_by_name(periodic, "dihedral").usesPeriodicBoundaryConditions()


def test_common_periodic_positions_guarantee_requested_padding():
    pdb = app.PDBFile(str(data_path("examples/2ntCG_cg_vs_conect.pdb")))
    coordinates = pdb.positions.value_in_unit(unit.nanometer)
    skewed = np.vstack([coordinates, coordinates[0], coordinates[0]]) * unit.nanometer
    padding_nm = 2.5

    (centered,) = prepare_common_periodic_positions(
        pdb.topology,
        (skewed,),
        padding_nm * unit.nanometer,
    )

    centered_nm = centered.value_in_unit(unit.nanometer)
    box_size_nm = pdb.topology.getUnitCellDimensions().value_in_unit(unit.nanometer)[0]
    tolerance = 1.0e-12
    assert float(np.min(centered_nm)) >= padding_nm - tolerance
    assert float(np.max(centered_nm)) <= box_size_nm - padding_nm + tolerance


def test_periodic_box_cutoff_validation_rejects_too_small_box():
    pdb = app.PDBFile(str(data_path("examples/2ntCG_cg_vs_conect.pdb")))
    system = CranberryForceField().createSystem(
        pdb.topology,
        positions=pdb.positions,
        periodic=True,
        box_padding=2 * unit.nanometer,
    )
    with pytest.raises(ValueError, match="Periodic box is too small.*electrostatic.*box-padding"):
        validate_periodic_box_cutoffs(system)


def test_create_simulation_periodic_sets_box_and_positions():
    simulation = create_simulation(data_path("examples/2ntCG_cg_vs_conect.pdb"), periodic=True, box_padding=3 * unit.nanometer, platform="CPU")
    assert simulation.topology.getPeriodicBoxVectors() is not None
    assert simulation.system.getDefaultPeriodicBoxVectors() is not None

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
    assert args["schema_version"] == 1
    assert args["run_kind"] == "md"
    assert args["steps"] == 1
    assert result.checkpoint_interval == 10
    assert args["report_interval"] == 1
    assert args["checkpoint_interval"] == 10
    assert args["n_record"] == 1000
    assert args["model"] == "cranberry-v1-alpha.1"
    assert args["restart_from"] is None
    assert args["append_outputs"] is False
    assert "pdb_sha256" in args
    assert args["temperature_kelvin"] == pytest.approx(298)
    assert args["salt_millimolar"] == pytest.approx(150)
    assert args["timestep_femtosecond"] == pytest.approx(5)
    assert args["platform"] == "CPU"
    assert args["actual_platform"] == "CPU"
    assert args["periodic"] is False
    assert args["box_padding_nanometer"] == pytest.approx(2.0)
    assert args["enforce_periodic_output"] is False
    assert args["log_progress"] is True

    log_header = result.log_path.read_text().splitlines()[0]
    assert "Kinetic Energy" in log_header
    assert "Total Energy" in log_header
    assert "Elapsed Time" in log_header
    assert "Time Remaining" in log_header
    assert "Progress (%)" in log_header

    detailed = result.detailed_log_path.read_text().splitlines()
    assert detailed[0].startswith('#"Step","Time (ps)","Potential Energy (kJ/mole)","bond (kJ/mole)"')
    assert all(
        f'"{name} (kJ/mole)"' in detailed[0]
        for name in ("stacking35", "stacking55", "stacking33")
    )
    assert len(detailed) == 2
    validate_canonical_pdb(result.final_pdb_path).raise_for_errors()



def test_run_md_can_omit_log_progress(tmp_path):
    result = run_md(
        data_path("examples/2ntCG_cg_vs_conect.pdb"),
        steps=2,
        report_interval=1,
        output_dir=tmp_path,
        platform="CPU",
        log_progress=False,
    )

    log_lines = result.log_path.read_text().splitlines()
    assert "Progress (%)" not in log_lines[0]
    assert json.loads(result.args_path.read_text())["log_progress"] is False


def test_run_md_accepts_explicit_checkpoint_interval(tmp_path):
    result = run_md(
        data_path("examples/2ntCG_cg_vs_conect.pdb"),
        steps=1,
        report_interval=1,
        checkpoint_interval=1,
        output_dir=tmp_path,
        platform="CPU",
    )

    args = json.loads(result.args_path.read_text())
    assert result.report_interval == 1
    assert result.checkpoint_interval == 1
    assert args["checkpoint_interval"] == 1


def test_run_md_writes_minimization_report(tmp_path):
    result = run_md(
        data_path("examples/2ntCG_cg_vs_conect.pdb"),
        steps=1,
        report_interval=1,
        output_dir=tmp_path,
        platform="CPU",
        write_minimization_report=True,
    )

    assert result.minimization_report_path is not None
    report = json.loads(result.minimization_report_path.read_text())
    assert "before_kj_per_mol" in report
    assert "after_kj_per_mol" in report
    assert "force_groups_before_kj_per_mol" in report
    assert "force_groups_after_kj_per_mol" in report
    for key in (
        "force_groups_before_kj_per_mol",
        "force_groups_after_kj_per_mol",
    ):
        assert {
            "stacking35",
            "stacking55",
            "stacking33",
        } <= report[key].keys()

def test_run_md_archives_distinct_args_only(tmp_path):
    pdb = data_path("examples/2ntCG_cg_vs_conect.pdb")
    run_md(pdb, steps=1, report_interval=1, output_dir=tmp_path, platform="CPU")
    run_md(pdb, steps=1, report_interval=1, output_dir=tmp_path, platform="CPU")
    history_dir = tmp_path / "args_history"
    assert not history_dir.exists()

    run_md(pdb, steps=2, report_interval=1, output_dir=tmp_path, platform="CPU")
    archived = history_dir / "000001_args.json"
    assert archived.exists()
    archived_args = json.loads(archived.read_text())
    current_args = json.loads((tmp_path / "args.json").read_text())
    assert archived_args["steps"] == 1
    assert current_args["steps"] == 2


def test_run_md_restart_errors_on_incompatible_args(tmp_path):
    pdb = data_path("examples/2ntCG_cg_vs_conect.pdb")
    first = run_md(pdb, steps=1, report_interval=1, output_dir=tmp_path, platform="CPU")
    args_path = tmp_path / "args.json"
    args = json.loads(args_path.read_text())
    args["temperature_kelvin"] = 310.0
    args_path.write_text(json.dumps(args, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="temperature_kelvin"):
        run_md(
            pdb,
            steps=1,
            report_interval=1,
            output_dir=tmp_path,
            restart_from=first.checkpoint_path,
            platform="CPU",
        )


def test_run_md_restart_warns_on_platform_args_mismatch(tmp_path):
    pdb = data_path("examples/2ntCG_cg_vs_conect.pdb")
    first = run_md(pdb, steps=1, report_interval=1, output_dir=tmp_path, platform="CPU")
    args_path = tmp_path / "args.json"
    args = json.loads(args_path.read_text())
    args["platform"] = "Reference"
    args_path.write_text(json.dumps(args, indent=2, sort_keys=True) + "\n")

    with pytest.warns(RuntimeWarning, match="platform"):
        run_md(
            pdb,
            steps=1,
            report_interval=1,
            output_dir=tmp_path,
            restart_from=first.checkpoint_path,
            platform="CPU",
        )


def test_run_md_restart_errors_on_corrupt_args_json(tmp_path):
    pdb = data_path("examples/2ntCG_cg_vs_conect.pdb")
    first = run_md(pdb, steps=1, report_interval=1, output_dir=tmp_path, platform="CPU")
    (tmp_path / "args.json").write_text("not-json")

    with pytest.raises(ValueError, match="not valid JSON"):
        run_md(
            pdb,
            steps=1,
            report_interval=1,
            output_dir=tmp_path,
            restart_from=first.checkpoint_path,
            platform="CPU",
        )


def test_run_md_restarts_from_checkpoint_and_appends_outputs(tmp_path):
    first = run_md(
        data_path("examples/2ntCG_cg_vs_conect.pdb"),
        steps=1,
        report_interval=1,
        output_dir=tmp_path,
        platform="CPU",
    )
    initial_dcd_size = first.dcd_path.stat().st_size
    second = run_md(
        data_path("examples/2ntCG_cg_vs_conect.pdb"),
        steps=1,
        report_interval=1,
        output_dir=tmp_path,
        restart_from=first.checkpoint_path,
        platform="CPU",
    )

    args = json.loads(second.args_path.read_text())
    assert args["restart_from"] == str(first.checkpoint_path)
    assert args["append_outputs"] is True
    detailed = second.detailed_log_path.read_text().splitlines()
    assert [line.split(",", 1)[0] for line in detailed[1:]] == ["1", "2"]
    log_lines = second.log_path.read_text().splitlines()
    log_header = next(csv.reader([log_lines[0].lstrip("#")]))
    step_index = log_header.index("Step")
    log_rows = list(csv.reader(line for line in log_lines[1:] if not line.startswith("#")))
    assert [row[step_index] for row in log_rows] == ["1", "2"]
    assert second.dcd_path.stat().st_size > initial_dcd_size


def test_run_md_restart_skips_minimization(tmp_path, monkeypatch):
    pdb = data_path("examples/2ntCG_cg_vs_conect.pdb")
    first = run_md(pdb, steps=1, report_interval=1, output_dir=tmp_path, platform="CPU")

    def fail_minimization(*args, **kwargs):
        raise AssertionError("restart should continue from checkpoint without minimization")

    monkeypatch.setattr(md_module, "_minimize_and_report", fail_minimization)
    second = run_md(
        pdb,
        steps=1,
        report_interval=1,
        output_dir=tmp_path,
        restart_from=first.checkpoint_path,
        platform="CPU",
        write_minimization_report=True,
    )

    args = json.loads(second.args_path.read_text())
    assert second.minimization_report_path is None
    assert args["write_minimization_report"] is False


def test_run_md_restart_missing_outputs_warns_and_creates_files(tmp_path):
    first = run_md(
        data_path("examples/2ntCG_cg_vs_conect.pdb"),
        steps=1,
        report_interval=1,
        output_dir=tmp_path / "first",
        platform="CPU",
    )
    with pytest.warns(RuntimeWarning) as warnings_record:
        second = run_md(
            data_path("examples/2ntCG_cg_vs_conect.pdb"),
            steps=1,
            report_interval=1,
            output_dir=tmp_path / "second",
            restart_from=first.checkpoint_path,
            platform="CPU",
        )
    warning_text = "\n".join(str(item.message) for item in warnings_record)
    assert "Restart output files are missing" in warning_text
    assert "Restart compatibility cannot be checked" in warning_text

    assert second.dcd_path.exists()
    assert second.log_path.exists()
    assert second.detailed_log_path.exists()
    detailed = second.detailed_log_path.read_text().splitlines()
    assert detailed[1].split(",", 1)[0] == "2"


def test_run_md_restart_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_md(
            data_path("examples/2ntCG_cg_vs_conect.pdb"),
            steps=1,
            output_dir=tmp_path,
            restart_from=tmp_path / "missing.chk",
            platform="CPU",
        )


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
