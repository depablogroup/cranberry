#!/usr/bin/env python
"""Run NVE near native speed, then rewind and localize the first instability."""

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
from nve_failure_diagnostic import (
    build_geometry_spec,
    force_group_energies,
    inspect_state,
    remove_com_velocity,
    write_json,
)

TEMPERATURE_K = 298.0
TIMESTEP_FS = 3.0
OUTPUT = Path("/tmp/cranberry-nve-adaptive-3fs")
TOTAL_STEPS = math.ceil(1_000_000_000 / TIMESTEP_FS)
COARSE_STEPS = round(1_000_000 / TIMESTEP_FS)  # 1 ns
MEDIUM_STEPS = round(1_000 / TIMESTEP_FS)  # 1 ps


def create_system_and_start():
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
    context = mm.Context(system, thermal, platform, properties)
    context.setPositions(positions)
    context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    mm.LocalEnergyMinimizer.minimize(context)
    context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin, 20260810)
    thermal.step(10000)
    state = context.getState(getPositions=True, getVelocities=True)
    start = {
        "positions": state.getPositions(),
        "velocities": remove_com_velocity(state.getVelocities(), system),
        "box": state.getPeriodicBoxVectors(),
    }
    del context, thermal
    return pdb, system, platform, properties, start


def create_nve_context(system, platform, properties, start=None, checkpoint=None):
    integrator = mm.VerletIntegrator(TIMESTEP_FS * unit.femtosecond)
    context = mm.Context(system, integrator, platform, properties)
    if checkpoint is not None:
        context.loadCheckpoint(checkpoint)
    else:
        context.setPositions(start["positions"])
        context.setVelocities(start["velocities"])
        context.setPeriodicBoxVectors(*start["box"])
    return context, integrator


def record(context, system, geometry, step, baseline):
    state = context.getState(getEnergy=True, getPositions=True, getForces=True)
    potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    total = potential + kinetic
    return {
        "step": step,
        "time_ps": step * TIMESTEP_FS / 1000,
        "potential_kj_per_mol": potential,
        "kinetic_kj_per_mol": kinetic,
        "total_kj_per_mol": total,
        "total_delta_kj_per_mol": total - baseline,
        **inspect_state(state, *geometry),
        "force_group_kj_per_mol": force_group_energies(context, system),
    }


def suspicious(item):
    scalars = [value for value in item.values() if isinstance(value, (int, float))]
    return (
        not np.all(np.isfinite(scalars))
        or item["max_bond_stretch_nm"] > 0.1
        or item["max_force_kj_per_mol_nm"] > 1e6
        or item["minimum_angle_sine"] < 1e-4
    )


def replay(system, platform, properties, geometry, checkpoint, start_step, baseline):
    context, integrator = create_nve_context(
        system, platform, properties, checkpoint=checkpoint
    )
    medium = []
    step = start_step
    previous_checkpoint = context.createCheckpoint()
    previous_step = step
    for _ in range(math.ceil(COARSE_STEPS / MEDIUM_STEPS)):
        previous_checkpoint = context.createCheckpoint()
        previous_step = step
        integrator.step(MEDIUM_STEPS)
        step += MEDIUM_STEPS
        item = record(context, system, geometry, step, baseline)
        medium.append(item)
        if suspicious(item):
            break
    del context, integrator

    context, integrator = create_nve_context(
        system, platform, properties, checkpoint=previous_checkpoint
    )
    fine = []
    step = previous_step
    for _ in range(MEDIUM_STEPS * 4):
        try:
            integrator.step(1)
            step += 1
            item = record(context, system, geometry, step, baseline)
            fine.append(item)
            if not np.isfinite(item["total_kj_per_mol"]):
                break
        except Exception as exc:
            fine.append({"step": step + 1, "error": f"{type(exc).__name__}: {exc}"})
            break
    return medium, fine


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pdb, system, platform, properties, start = create_system_and_start()
    geometry = build_geometry_spec(system, pdb.topology)
    context, integrator = create_nve_context(system, platform, properties, start=start)
    initial = context.getState(getEnergy=True)
    baseline = (
        initial.getPotentialEnergy() + initial.getKineticEnergy()
    ).value_in_unit(unit.kilojoule_per_mole)
    initial_checkpoint = context.createCheckpoint()
    (OUTPUT / "checkpoint-0000ns.chk").write_bytes(initial_checkpoint)

    coarse = []
    elapsed = 0
    previous_checkpoint = initial_checkpoint
    previous_step = 0
    started = time.time()
    status = "completed"
    error = None
    medium = []
    fine = []
    while elapsed < TOTAL_STEPS:
        block = min(COARSE_STEPS, TOTAL_STEPS - elapsed)
        try:
            integrator.step(block)
            elapsed += block
            item = record(context, system, geometry, elapsed, baseline)
            coarse.append(item)
            checkpoint = context.createCheckpoint()
            ns_index = round(elapsed * TIMESTEP_FS / 1_000_000)
            (OUTPUT / f"checkpoint-{ns_index:04d}ns.chk").write_bytes(checkpoint)
            write_json(OUTPUT / "progress.json", {
                "status": "running", "elapsed_steps": elapsed, "coarse": coarse
            })
            print(f"{elapsed}/{TOTAL_STEPS} ({elapsed * TIMESTEP_FS / 1e6:.3f} ns)", flush=True)
            if suspicious(item):
                status = "localized"
                medium, fine = replay(
                    system, platform, properties, geometry,
                    previous_checkpoint, previous_step, baseline,
                )
                break
            previous_checkpoint = checkpoint
            previous_step = elapsed
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            medium, fine = replay(
                system, platform, properties, geometry,
                previous_checkpoint, previous_step, baseline,
            )
            break
    result = {
        "status": status,
        "error": error,
        "elapsed_steps": elapsed,
        "elapsed_ns": elapsed * TIMESTEP_FS / 1e6,
        "wall_seconds": time.time() - started,
        "baseline_total_kj_per_mol": baseline,
        "coarse": coarse,
        "medium": medium,
        "fine": fine,
    }
    write_json(OUTPUT / "result.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"coarse", "medium", "fine"}}))


if __name__ == "__main__":
    main()
