#!/usr/bin/env python
"""Descend through NVE timesteps until 1l2x survives a full microsecond."""

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
from nve_failure_diagnostic import remove_com_velocity, write_json

OUTPUT = Path("/tmp/cranberry-nve-survival-bracket.json")
TEMPERATURE_K = 298.0
TIMESTEPS_FS = (2.5, 2.25, 2.0, 1.5, 1.0)


def initial_state():
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
    integrator = mm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin, friction, 1 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(20260811)
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions(positions)
    context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    mm.LocalEnergyMinimizer.minimize(context)
    context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin, 20260810)
    integrator.step(10000)
    state = context.getState(getPositions=True, getVelocities=True)
    start = {
        "positions": state.getPositions(),
        "velocities": remove_com_velocity(state.getVelocities(), system),
        "box": state.getPeriodicBoxVectors(),
    }
    del context, integrator
    return system, platform, properties, start


def run(system, platform, properties, start, timestep_fs):
    integrator = mm.VerletIntegrator(timestep_fs * unit.femtosecond)
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions(start["positions"])
    context.setVelocities(start["velocities"])
    context.setPeriodicBoxVectors(*start["box"])
    total_steps = math.ceil(1_000_000_000 / timestep_fs)
    block_steps = round(1_000_000 / timestep_fs)
    elapsed = 0
    records = []
    status = "completed"
    error = None
    started = time.time()
    try:
        while elapsed < total_steps:
            block = min(block_steps, total_steps - elapsed)
            integrator.step(block)
            elapsed += block
            state = context.getState(getEnergy=True)
            potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            total = potential + kinetic
            if not np.isfinite(total):
                raise FloatingPointError(f"non-finite energy at step {elapsed}")
            records.append({
                "step": elapsed,
                "time_ns": elapsed * timestep_fs / 1_000_000,
                "potential_kj_per_mol": potential,
                "kinetic_kj_per_mol": kinetic,
                "total_kj_per_mol": total,
            })
            if len(records) % 10 == 0:
                print(f"dt={timestep_fs:g} fs: {records[-1]['time_ns']:.1f} ns", flush=True)
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    return {
        "timestep_fs": timestep_fs,
        "status": status,
        "error": error,
        "elapsed_steps": elapsed,
        "elapsed_ns": elapsed * timestep_fs / 1_000_000,
        "wall_seconds": time.time() - started,
        "records": records,
    }


def main():
    system, platform, properties, start = initial_state()
    payload = {"target_time_us": 1.0, "runs": []}
    for timestep_fs in TIMESTEPS_FS:
        row = run(system, platform, properties, start, timestep_fs)
        payload["runs"].append(row)
        write_json(OUTPUT, payload)
        print(json.dumps({key: value for key, value in row.items() if key != "records"}))
        if row["status"] == "completed":
            break


if __name__ == "__main__":
    main()
