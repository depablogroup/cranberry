#!/usr/bin/env python
"""Compare constrained Langevin velocity statistics with canonical predictions."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
from openmm import app, unit
from scipy.stats import chi2, norm

from cranberry.data import data_path
from cranberry.forcefield import CranberryForceField, prepare_periodic_positions
from cranberry.md import calculate_langevin_friction

SYSTEMS = ("2ntCG", "1l2x")
TEMPERATURE_K = 298.0
R = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)


def sample(system_id: str, production_ps: float = 100.0):
    pdb = app.PDBFile(str(data_path(f"examples/{system_id}_cg_vs_conect.pdb")))
    positions = prepare_periodic_positions(pdb.topology, pdb.positions, 4 * unit.nanometer)
    system = CranberryForceField().createSystem(
        pdb.topology, positions=positions, temperature=TEMPERATURE_K * unit.kelvin,
        periodic=True, box_padding=4 * unit.nanometer,
    )
    friction = calculate_langevin_friction(pdb.topology, system, TEMPERATURE_K * unit.kelvin)
    integrator = mm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin, friction, 8 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(20260812)
    context = mm.Context(
        system, integrator, mm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"}
    )
    context.setPositions(positions)
    context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    mm.LocalEnergyMinimizer.minimize(context)
    context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin, 20260812)
    integrator.step(12500)

    masses = np.array([
        system.getParticleMass(i).value_in_unit(unit.dalton)
        for i in range(system.getNumParticles())
    ])
    massive = masses > 0
    n_massive = int(massive.sum())
    dof = 3 * n_massive - system.getNumConstraints() - 3
    projected_scale = np.sqrt(dof / (3 * n_massive))
    velocities = []
    kinetic = []
    samples = int(round(production_ps))
    for _ in range(samples):
        integrator.step(125)
        state = context.getState(getEnergy=True, getVelocities=True)
        values = state.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        )
        com = np.average(values[massive], axis=0, weights=masses[massive])
        z = (
            (values[massive] - com)
            / np.sqrt(R * TEMPERATURE_K / masses[massive])[:, None]
            / projected_scale
        )
        velocities.append(z.ravel())
        energy = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        kinetic.append(2 * energy / (R * TEMPERATURE_K))
    return np.concatenate(velocities), np.asarray(kinetic), dof, system.getNumConstraints()


def main():
    output_dir = Path("docs/dev/progress")
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    summary = {}
    for column, system_id in enumerate(SYSTEMS):
        z, kinetic, dof, constraints = sample(system_id)
        grid = np.linspace(-4, 4, 500)
        axes[0, column].hist(z, bins=60, density=True, alpha=0.55, label="OpenMM 8 fs")
        axes[0, column].plot(grid, norm.pdf(grid), linewidth=2, label="projected Gaussian")
        axes[0, column].set_title(f"{system_id}: velocity components")
        axes[0, column].set_xlabel("variance-corrected standardized velocity")
        axes[0, column].set_ylabel("density")
        axes[0, column].legend()

        low, high = chi2.ppf([0.001, 0.999], dof)
        grid = np.linspace(low, high, 500)
        axes[1, column].hist(kinetic, bins=40, density=True, alpha=0.55, label="OpenMM 8 fs")
        axes[1, column].plot(grid, chi2.pdf(grid, dof), linewidth=2, label=f"chi-square({dof})")
        axes[1, column].set_title(f"{system_id}: kinetic energy")
        axes[1, column].set_xlabel("2K/(RT)")
        axes[1, column].set_ylabel("density")
        axes[1, column].legend()
        centered = z - z.mean()
        summary[system_id] = {
            "constraints": constraints,
            "degrees_of_freedom": dof,
            "velocity_samples": int(z.size),
            "velocity_mean": float(z.mean()),
            "velocity_std": float(z.std()),
            "velocity_skew": float(np.mean(centered**3) / z.std() ** 3),
            "velocity_excess_kurtosis": float(np.mean(centered**4) / z.std() ** 4 - 3),
            "kinetic_x_mean": float(kinetic.mean()),
            "kinetic_x_variance": float(kinetic.var()),
            "chi2_expected_mean": dof,
            "chi2_expected_variance": 2 * dof,
        }
    figure.suptitle("8 fs Langevin velocity distributions at 298 K")
    figure.savefig(output_dir / "velocity-distribution-8fs.png", dpi=180)
    (output_dir / "velocity-distribution-8fs.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
