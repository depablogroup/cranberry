import json

import pytest
from openmm import app, unit

from cranberry.data import data_path
from cranberry.forcefield import CranberryForceField
from cranberry.md import calculate_langevin_friction, create_simulation, run_md
from cranberry.validation import validate_canonical_pdb


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
    assert args["report_interval"] == 1
    assert args["model"] == "cranberry-v1-alpha.1"
    assert args["restart_from"] is None
    assert args["append_outputs"] is False
    assert "pdb_sha256" in args
    assert args["temperature_kelvin"] == pytest.approx(298)
    assert args["salt_millimolar"] == pytest.approx(150)
    assert args["timestep_femtosecond"] == pytest.approx(10)

    detailed = result.detailed_log_path.read_text().splitlines()
    assert detailed[0].startswith('#"Step","Time (ps)","Potential Energy (kJ/mole)","bond (kJ/mole)"')
    assert len(detailed) == 2
    validate_canonical_pdb(result.final_pdb_path).raise_for_errors()


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
    log_lines = [line for line in second.log_path.read_text().splitlines() if not line.startswith("#")]
    assert [line.split(",", 1)[0] for line in log_lines] == ["1", "2"]
    assert second.dcd_path.stat().st_size > initial_dcd_size


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
