from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openmm import LangevinMiddleIntegrator, Platform, unit
from openmm import app

from cranberry.forcefield import FORCE_GROUP_IDS, CranberryForceField
from cranberry.validation import validate_canonical_pdb


@dataclass(frozen=True)
class EnergyReport:
    total_potential_energy: unit.Quantity
    components: dict[str, unit.Quantity]

    def as_kj_per_mol(self) -> dict[str, float]:
        values = {
            "total": self.total_potential_energy.value_in_unit(unit.kilojoule_per_mole),
        }
        values.update({name: energy.value_in_unit(unit.kilojoule_per_mole) for name, energy in self.components.items()})
        return values


def compute_energy(
    pdb_path: str | Path,
    *,
    model: str = "default",
    temperature=298 * unit.kelvin,
    salt_concentration=150 * unit.millimolar,
    platform: str | None = "CPU",
) -> EnergyReport:
    """Compute total and force-group energies for a canonical CRANBERRY CG PDB."""

    validation = validate_canonical_pdb(pdb_path)
    validation.raise_for_errors()

    pdb = app.PDBFile(str(pdb_path))
    forcefield = CranberryForceField(model)
    system = forcefield.createSystem(
        pdb.topology,
        positions=pdb.positions,
        temperature=temperature,
        salt_concentration=salt_concentration,
    )
    integrator = LangevinMiddleIntegrator(temperature, 1 / unit.picosecond, 1 * unit.femtosecond)
    if platform is None:
        simulation = app.Simulation(pdb.topology, system, integrator)
    else:
        simulation = app.Simulation(pdb.topology, system, integrator, Platform.getPlatformByName(platform))
    simulation.context.setPositions(pdb.positions)

    total = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    components = {}
    for name, group in FORCE_GROUP_IDS.items():
        if group >= 32:
            continue
        energy = simulation.context.getState(getEnergy=True, groups={group}).getPotentialEnergy()
        if abs(energy.value_in_unit(unit.kilojoule_per_mole)) > 0 or name in _present_force_names(system):
            components[name] = energy
    return EnergyReport(total_potential_energy=total, components=components)


def _present_force_names(system) -> set[str]:
    return {system.getForce(i).getName() for i in range(system.getNumForces())}
