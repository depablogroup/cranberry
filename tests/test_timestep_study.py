import json

import numpy as np
import openmm as mm
import pytest
from openmm import app, unit

from benchmarks.timestep_study import (
    degrees_of_freedom,
    geometry_observable_spec,
    geometry_snapshot,
    summarize,
    write_json_atomic,
)
from cranberry.data import data_path
from cranberry.forcefield import CranberryForceField


def test_degrees_of_freedom_tracks_explicit_com_removal():
    system = mm.System()
    system.addParticle(1)
    system.addParticle(1)
    system.addParticle(0)
    system.addConstraint(0, 1, 0.1)

    assert degrees_of_freedom(system) == 5
    assert degrees_of_freedom(system, remove_com=True) == 2


def test_summarize_reports_normalized_linear_nve_drift():
    system = mm.System()
    system.addParticle(1)
    system.addParticle(1)
    samples = {
        "time_ps": [1.0, 2.0, 3.0, 4.0],
        "total_kj_per_mol": [10.0, 10.1, 10.2, 10.3],
    }

    summary = summarize(samples, system, 300.0, "nve")

    assert summary["n_samples"] == 4
    assert summary["energy_drift_kj_per_mol_per_ns"] == pytest.approx(100.0)
    assert summary["energy_drift_se_kj_per_mol_per_ns"] == pytest.approx(0.0, abs=1e-10)
    assert summary["energy_rms_fluctuation_kbt_per_dof"] > 0


def test_geometry_snapshot_records_labeled_local_pucker_observables():
    pdb = app.PDBFile(str(data_path("examples/2ntCG_cg_vs_conect.pdb")))
    forcefield = CranberryForceField()
    system = forcefield.createSystem(pdb.topology, positions=pdb.positions)
    spec = geometry_observable_spec(system, pdb.topology, forcefield)

    values = geometry_snapshot(
        pdb.positions, system.getDefaultPeriodicBoxVectors(), spec
    )

    for name, observed in values.items():
        assert len(observed) == len(spec["labels"][name])
    assert len(values["s2_b1_lengths_nm"]) == 2
    assert len(values["p_s3_lengths_nm"]) == 1
    assert len(values["pucker_phase_degrees"]) == 1
    assert np.isfinite(values["pucker_phase_degrees"][0])



def test_geometry_phase_matches_openmm_force_expression():
    pdb = app.PDBFile(str(data_path("examples/2ntCG_cg_vs_conect.pdb")))
    forcefield = CranberryForceField()
    source_system = forcefield.createSystem(pdb.topology, positions=pdb.positions)
    spec = geometry_observable_spec(source_system, pdb.topology, forcefield)
    phase_force = mm.CustomCompoundBondForce(6, forcefield._phase_angle_expression(spec["descriptors"]))
    for index, weight in enumerate(spec["weights"]):
        phase_force.addGlobalParameter(f"w{index}", float(weight))
    phase_force.addGlobalParameter("intercept", spec["intercept"])
    phase_force.addBond(spec["pucker_groups"][0])
    phase_system = mm.System()
    for _ in pdb.topology.atoms():
        phase_system.addParticle(1.0)
    phase_system.addForce(phase_force)
    integrator = mm.VerletIntegrator(1 * unit.femtosecond)
    context = mm.Context(phase_system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(pdb.positions)

    expected = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    observed = geometry_snapshot(
        pdb.positions, source_system.getDefaultPeriodicBoxVectors(), spec
    )["pucker_phase_degrees"][0]

    assert observed == pytest.approx(expected, abs=1e-10)



def test_write_json_atomic_replaces_snapshot(tmp_path):
    path = tmp_path / "study.json"

    write_json_atomic(path, {"run": 1})
    write_json_atomic(path, {"run": 2})

    assert json.loads(path.read_text()) == {"run": 2}
    assert not path.with_suffix(".json.tmp").exists()
