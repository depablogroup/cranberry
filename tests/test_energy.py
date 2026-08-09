import pytest

from cranberry.data import data_path
from cranberry.energy import compute_energy
from cranberry.energy_decomposition import present_force_group_names
from cranberry.forcefield import (
    FUSED_STACKING_FORCE_NAME,
    STACKING_SCALE_PARAMETERS,
    CranberryForceField,
)
from openmm import app


# Generated with the GPU workshop copy of OpenMM-CGRNA using
# run_rna.py --dry-run --post-run-reporting --use-pdb, pucker-mode all,
# base-form aniso, high-resolution spline, cranberry0.2.19.h5, and
# angle-scaling 0.1. For Cranberry, pucker includes legacy pucker + terminal_U3.
EXPECTED_ENERGIES = {
    "2ntCG_cg_vs_conect.pdb": {
        "total": 29.685498622276896,
        "bond": 17.462527361998394,
        "angle": 0.796891488925228,
        "dihedral": 3.693461288052711,
        "pucker": 12.793714162224834,
        "stacking35": -6.833475903545025e-22,
        "stacking55": 0.0,
        "stacking33": 0.0,
        "pairing": 0.0,
        "wca": 0.0,
        "spline": -5.061095678924274,
        "electrostatic": 0.0,
    },
    "157d_cg_vs_conect.pdb": {
        "total": -408.2543522195409,
        "bond": 73.61284323637926,
        "angle": 6.158542788801296,
        "dihedral": 99.65224273030286,
        "pucker": 37.62409940584949,
        "stacking35": -143.2802892279098,
        "stacking55": -47.81499746159594,
        "stacking33": -9.893400525737245e-46,
        "pairing": -310.6757454248722,
        "wca": 130.9731436211584,
        "spline": -265.29367221148993,
        "electrostatic": 10.789480323835647,
    },
    "1l2x_cg_vs_conect.pdb": {
        "total": 854.1646598337904,
        "bond": 382.5820370565472,
        "angle": 13.606584351844113,
        "dihedral": 1089.5108475398738,
        "pucker": 60.898531981023496,
        "stacking35": -139.29861093076445,
        "stacking55": -71.99834779067059,
        "stacking33": -7.214988294378932e-32,
        "pairing": -283.6257210398419,
        "wca": 11.934818048272357,
        "spline": -227.885289637754,
        "electrostatic": 18.43981025526036,
    },
    "1zih_cg_vs_conect.pdb": {
        "total": 171.1675562407328,
        "bond": 67.34505012462733,
        "angle": 5.919482619553215,
        "dihedral": 303.6612371036814,
        "pucker": 14.10139187935362,
        "stacking35": -91.12990724324229,
        "stacking55": -0.1550417822287445,
        "stacking33": -5.837238320588313e-40,
        "pairing": -141.79771719033235,
        "wca": 111.37827243358372,
        "spline": -101.76517184857472,
        "electrostatic": 3.609960144311606,
    },
}



def test_create_system_adds_named_forces():
    pdb = app.PDBFile(str(data_path("examples/2ntCG_cg_vs_conect.pdb")))
    system = CranberryForceField().createSystem(pdb.topology, positions=pdb.positions)
    names = {system.getForce(index).getName() for index in range(system.getNumForces())}
    assert {"bond", "angle", "dihedral", "pucker", "stacking", "pairing", "wca", "spline", "electrostatic"} <= names
    spline = next(force for force in (system.getForce(i) for i in range(system.getNumForces())) if force.getName() == "spline")
    assert spline.getNumTabulatedFunctions() == 3
    assert {spline.getTabulatedFunctionName(i) for i in range(3)} == {"U", "rmin", "rmax"}
    stacking = next(
        force
        for force in (
            system.getForce(i) for i in range(system.getNumForces())
        )
        if force.getName() == FUSED_STACKING_FORCE_NAME
    )
    global_parameters = {
        stacking.getGlobalParameterName(i)
        for i in range(stacking.getNumGlobalParameters())
    }
    assert global_parameters == set(STACKING_SCALE_PARAMETERS.values())


def test_pairing_is_packed_by_donor_type():
    pdb = app.PDBFile(str(data_path("examples/1l2x_cg_vs_conect.pdb")))
    system = CranberryForceField().createSystem(
        pdb.topology, positions=pdb.positions, enabled_forces=["pairing"]
    )
    pairing_forces = [
        system.getForce(index)
        for index in range(system.getNumForces())
        if system.getForce(index).getName() == "pairing"
    ]

    assert len(pairing_forces) == 4
    assert all(force.getNumPerAcceptorParameters() >= 14 for force in pairing_forces)
    assert all(force.getNumPerAcceptorParameters() % 14 == 0 for force in pairing_forces)
    assert all(force.getNumTabulatedFunctions() == 0 for force in pairing_forces)


def test_fused_stacking_preserves_enabled_component_selection():
    pdb = app.PDBFile(str(data_path("examples/1l2x_cg_vs_conect.pdb")))
    system = CranberryForceField().createSystem(
        pdb.topology,
        positions=pdb.positions,
        enabled_forces=["stacking55"],
    )
    stacking = next(
        force
        for force in (
            system.getForce(i) for i in range(system.getNumForces())
        )
        if force.getName() == FUSED_STACKING_FORCE_NAME
    )

    assert present_force_group_names(system) == ["stacking55"]
    assert stacking.getNumGlobalParameters() == 1
    assert (
        stacking.getGlobalParameterName(0)
        == STACKING_SCALE_PARAMETERS["stacking55"]
    )


@pytest.mark.parametrize("filename,expected", EXPECTED_ENERGIES.items())
def test_energy_regression_cpu(filename, expected):
    report = compute_energy(data_path(f"examples/{filename}"), platform="CPU")
    values = report.as_kj_per_mol()
    for name, value in expected.items():
        assert values[name] == pytest.approx(value, abs=1e-6)
