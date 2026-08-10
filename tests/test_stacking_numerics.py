import numpy as np
import pytest
import openmm as mm
from openmm import app, unit

from cranberry.data import data_path
from cranberry.forcefield import CranberryForceField


LEGACY_COS_NORMAL_PSI = (
    "select(sin(A)*sin(B), full, partial); "
    "full=sin(A)*sin(B)*cos(PHI)-cos(A)*cos(B); "
    "partial=-cos(A)*cos(B); "
    "A=angle(d2,d1,a1); B=angle(d1,a1,a2); "
    "PHI=dihedral(d2,d1,a1,a2)"
)

DISTANCE_COS_NORMAL_PSI = (
    "(distance(d2,a1)^2+distance(d1,a2)^2-distance(d2,a2)^2-distance(d1,a1)^2)"
    "/(2*distance(d1,d2)*distance(a1,a2))"
)


def _hbond_force(expression: str) -> mm.CustomHbondForce:
    force = mm.CustomHbondForce(expression)
    force.addDonor(1, 0, -1, [])
    force.addAcceptor(2, 3, -1, [])
    return force


def _evaluate_hbond_expression(expression: str, coordinates_nm: np.ndarray):
    system = mm.System()
    for _ in range(4):
        system.addParticle(1.0 * unit.dalton)
    system.addForce(_hbond_force(expression))
    platform = mm.Platform.getPlatformByName("CPU")
    context = mm.Context(system, mm.VerletIntegrator(0.001 * unit.picoseconds), platform)
    context.setPositions(unit.Quantity(coordinates_nm, unit.nanometer))
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    del context
    return energy, forces


def test_distance_cos_normal_matches_legacy_formula_off_singularity():
    coordinates_nm = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.4, 0.5],
            [1.1, 0.2, 0.1],
            [1.5, 0.8, -0.2],
        ],
        dtype=float,
    )

    legacy_energy, legacy_forces = _evaluate_hbond_expression(LEGACY_COS_NORMAL_PSI, coordinates_nm)
    distance_energy, distance_forces = _evaluate_hbond_expression(DISTANCE_COS_NORMAL_PSI, coordinates_nm)

    assert distance_energy == pytest.approx(legacy_energy, abs=1e-14)
    np.testing.assert_allclose(distance_forces, legacy_forces, atol=1e-12, rtol=1e-12)


def test_distance_cos_normal_keeps_collinear_stacking_cpu_forces_finite():
    coordinates_nm = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    expression = (
        "g1*g2*g3; "
        "g1=1/2*(tanh(theta_sens*(-cos(D2D1A1)-cos(theta0)))+1); "
        "g2=1/2*(tanh(theta_sens*(cos(D1A1A2)-cos(theta0)))+1); "
        "g3=1/2*(tanh(theta_sens*(cos_normal_psi-cos(psi0)))+1); "
        f"cos_normal_psi={DISTANCE_COS_NORMAL_PSI}; "
        "D2D1A1=angle(d2,d1,a1); D1A1A2=angle(d1,a1,a2); "
        "theta_sens=10; theta0=2.2; psi0=2.4"
    )

    energy, forces = _evaluate_hbond_expression(expression, coordinates_nm)

    assert np.isfinite(energy)
    assert np.isfinite(forces).all()

    system = mm.System()
    for _ in range(4):
        system.addParticle(1.0 * unit.dalton)
    system.addForce(_hbond_force(expression))
    context = mm.Context(
        system,
        mm.VerletIntegrator(0.001 * unit.picoseconds),
        mm.Platform.getPlatformByName("CPU"),
    )
    context.setPositions(unit.Quantity(coordinates_nm, unit.nanometer))
    mm.LocalEnergyMinimizer.minimize(context, maxIterations=1)
    del context


def test_cranberry_stacking35_cpu_forces_finite_for_collinear_base_normals():
    pdb = app.PDBFile(str(data_path("examples/2ntCG_cg_vs_conect.pdb")))
    coordinates_nm = np.array(
        [position.value_in_unit(unit.nanometer) for position in pdb.positions],
        dtype=float,
    )
    virtual_indices = {
        (atom.residue.index, atom.name): atom.index
        for atom in pdb.topology.atoms()
        if atom.name in {"BC", "BN"}
    }
    coordinates_nm[virtual_indices[(0, "BC")]] = [0.0, 0.0, 0.0]
    coordinates_nm[virtual_indices[(0, "BN")]] = [-1.0, 0.0, 0.0]
    coordinates_nm[virtual_indices[(1, "BC")]] = [1.0, 0.0, 0.0]
    coordinates_nm[virtual_indices[(1, "BN")]] = [2.0, 0.0, 0.0]
    positions = unit.Quantity(coordinates_nm, unit.nanometer)
    system = CranberryForceField().createSystem(
        pdb.topology,
        positions=positions,
        enabled_forces=["stacking35"],
    )
    context = mm.Context(
        system,
        mm.VerletIntegrator(0.001 * unit.picoseconds),
        mm.Platform.getPlatformByName("CPU"),
    )
    context.setPositions(positions)

    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

    assert np.isfinite(energy)
    assert np.isfinite(forces).all()
    mm.LocalEnergyMinimizer.minimize(context, maxIterations=1)
    del context
