#!/usr/bin/env python
"""Localize long-time NVE instability with persistent geometry diagnostics."""

import json
import math
import time
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit

from cranberry.data import data_path
from cranberry.forcefield import CranberryForceField, prepare_periodic_positions
from cranberry.md import calculate_langevin_friction

TEMPERATURE_K = 298.0
TIMESTEP_FS = 3.0
OUTPUT = Path("/tmp/cranberry-nve-diagnostic-3fs")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def remove_com_velocity(velocities, system):
    values = np.asarray(velocities.value_in_unit(unit.nanometer / unit.picosecond)).copy()
    masses = np.array([
        system.getParticleMass(index).value_in_unit(unit.dalton)
        for index in range(system.getNumParticles())
    ])
    massive = masses > 0
    values[massive] -= np.average(values[massive], axis=0, weights=masses[massive])
    return values * unit.nanometer / unit.picosecond


def build_geometry_spec(system, topology):
    atoms = list(topology.atoms())
    bonds = []
    triplets = []
    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            for index in range(force.getNumBonds()):
                first, second, equilibrium, stiffness = force.getBondParameters(index)
                if stiffness <= 0 * unit.kilojoule_per_mole / unit.nanometer**2:
                    continue
                first, second = int(first), int(second)
                bonds.append((
                    first,
                    second,
                    equilibrium.value_in_unit(unit.nanometer),
                    f"{atoms[first].residue.index}:{atoms[first].name}-"
                    f"{atoms[second].residue.index}:{atoms[second].name}",
                ))
        elif isinstance(force, mm.HarmonicAngleForce):
            for index in range(force.getNumAngles()):
                first, center, last, _, _ = force.getAngleParameters(index)
                triplets.append((int(first), int(center), int(last), f"angle:{index}"))
        elif isinstance(force, mm.CustomCompoundBondForce) and force.getName() in {"dihedral", "pucker"}:
            for index in range(force.getNumBonds()):
                particles, _ = force.getBondParameters(index)
                particles = [int(value) for value in particles]
                for offset in range(len(particles) - 2):
                    triplets.append((*particles[offset:offset + 3], f"{force.getName()}:{index}:{offset}"))
    return bonds, triplets


def inspect_state(state, bonds, triplets):
    positions = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    box = np.asarray(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
    inverse_box = np.linalg.inv(box)

    def displacement(first, second):
        delta = positions[second] - positions[first]
        return delta - np.rint(delta @ inverse_box) @ box

    stretches = []
    for first, second, equilibrium, label in bonds:
        length = float(np.linalg.norm(displacement(first, second)))
        stretches.append((abs(length - equilibrium), length, equilibrium, label, first, second))
    worst = max(stretches)

    conditioning = []
    for first, center, last, label in triplets:
        left = displacement(center, first)
        right = displacement(center, last)
        sine = np.linalg.norm(np.cross(left, right)) / (np.linalg.norm(left) * np.linalg.norm(right))
        conditioning.append((float(sine), label, first, center, last))
    minimum = min(conditioning)

    forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    ))
    norms = np.linalg.norm(forces, axis=1)
    return {
        "max_bond_stretch_nm": worst[0],
        "max_bond_length_nm": worst[1],
        "max_bond_equilibrium_nm": worst[2],
        "max_bond_label": worst[3],
        "max_bond_atoms": [worst[4], worst[5]],
        "minimum_angle_sine": minimum[0],
        "minimum_angle_sine_label": minimum[1],
        "minimum_angle_sine_atoms": list(minimum[2:]),
        "max_force_kj_per_mol_nm": float(norms.max()),
        "max_force_atom": int(norms.argmax()),
    }


def force_group_energies(context, system):
    energies = {}
    for force in system.getForces():
        name = force.getName() or type(force).__name__
        state = context.getState(getEnergy=True, groups={force.getForceGroup()})
        energies[name] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return energies


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pdb = app.PDBFile(str(data_path("examples/1l2x_cg_vs_conect.pdb")))
    positions = prepare_periodic_positions(pdb.topology, pdb.positions, 4 * unit.nanometer)
    system = CranberryForceField().createSystem(
        pdb.topology,
        positions=positions,
        temperature=TEMPERATURE_K * unit.kelvin,
        periodic=True,
        box_padding=4 * unit.nanometer,
    )
    platform = mm.Platform.getPlatformByName("CUDA")
    properties = {"Precision": "mixed"}
    friction = calculate_langevin_friction(pdb.topology, system, TEMPERATURE_K * unit.kelvin)

    thermal = mm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin, friction, 1 * unit.femtosecond
    )
    thermal.setRandomNumberSeed(20260811)
    thermal_context = mm.Context(system, thermal, platform, properties)
    thermal_context.setPositions(positions)
    thermal_context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    mm.LocalEnergyMinimizer.minimize(thermal_context)
    thermal_context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin, 20260810)
    thermal.step(10000)
    start = thermal_context.getState(getPositions=True, getVelocities=True)

    integrator = mm.VerletIntegrator(TIMESTEP_FS * unit.femtosecond)
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions(start.getPositions())
    context.setPeriodicBoxVectors(*start.getPeriodicBoxVectors())
    context.setVelocities(remove_com_velocity(start.getVelocities(), system))
    del thermal_context, thermal

    bonds, triplets = build_geometry_spec(system, pdb.topology)
    total_steps = math.ceil(1_000_000_000 / TIMESTEP_FS)
    sample_steps = round(10_000 / TIMESTEP_FS)
    checkpoint_steps = round(100_000 / TIMESTEP_FS)
    records = []
    elapsed = 0
    started = time.time()
    status = "completed"
    error = None
    initial_total = None
    try:
        while elapsed < total_steps:
            block = min(sample_steps, total_steps - elapsed)
            integrator.step(block)
            elapsed += block
            state = context.getState(
                getEnergy=True, getPositions=True, getVelocities=True, getForces=True
            )
            potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            total = potential + kinetic
            if initial_total is None:
                initial_total = total
            record = {
                "step": elapsed,
                "time_ps": elapsed * TIMESTEP_FS / 1000,
                "potential_kj_per_mol": potential,
                "kinetic_kj_per_mol": kinetic,
                "total_kj_per_mol": total,
                "total_delta_kj_per_mol": total - initial_total,
                **inspect_state(state, bonds, triplets),
            }
            if len(records) % 100 == 0 or record["max_bond_stretch_nm"] > 0.2:
                record["force_group_kj_per_mol"] = force_group_energies(context, system)
            scalar_values = [value for value in record.values() if isinstance(value, (int, float))]
            if not np.all(np.isfinite(scalar_values)):
                raise FloatingPointError(f"non-finite state at step {elapsed}")
            records.append(record)
            if elapsed % checkpoint_steps == 0 or elapsed == total_steps:
                (OUTPUT / "checkpoint.chk").write_bytes(context.createCheckpoint())
            if len(records) % 100 == 0 or elapsed == total_steps:
                write_json(OUTPUT / "progress.json", {
                    "status": "running", "elapsed_steps": elapsed, "records": records
                })
                print(f"{elapsed}/{total_steps} ({elapsed * TIMESTEP_FS / 1e6:.3f} ns)", flush=True)
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    result = {
        "status": status,
        "error": error,
        "elapsed_steps": elapsed,
        "elapsed_ns": elapsed * TIMESTEP_FS / 1e6,
        "wall_seconds": time.time() - started,
        "records": records,
    }
    write_json(OUTPUT / "result.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))


if __name__ == "__main__":
    main()
