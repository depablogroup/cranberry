#!/usr/bin/env python
"""Identify force terms responsible for short-time NVE energy drift."""

import json
import time
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit

from cranberry.data import data_path
from cranberry.forcefield import (
    FORCE_GROUP_NAMES,
    CranberryForceField,
    prepare_periodic_positions,
)
from cranberry.md import calculate_langevin_friction
from nve_failure_diagnostic import remove_com_velocity, write_json

OUTPUT = Path("/tmp/cranberry-nve-drift-ablation.json")
TEMPERATURE_K = 298.0
PRODUCTION_NS = 5.0

VARIANTS = {
    "full": set(FORCE_GROUP_NAMES),
    "no_electrostatic": set(FORCE_GROUP_NAMES) - {"electrostatic"},
    "no_spline": set(FORCE_GROUP_NAMES) - {"spline"},
    "no_wca": set(FORCE_GROUP_NAMES) - {"wca"},
    "no_pairing": set(FORCE_GROUP_NAMES) - {"pairing"},
    "no_stacking": set(FORCE_GROUP_NAMES) - {"stacking35", "stacking55", "stacking33"},
    "no_pucker": set(FORCE_GROUP_NAMES) - {"pucker"},
    "no_bond": set(FORCE_GROUP_NAMES) - {"bond"},
}


def run_condition(name, enabled, timestep_fs, periodic=True):
    pdb = app.PDBFile(str(data_path("examples/1l2x_cg_vs_conect.pdb")))
    positions = prepare_periodic_positions(pdb.topology, pdb.positions, 4 * unit.nanometer)
    system = CranberryForceField().createSystem(
        pdb.topology,
        positions=positions,
        temperature=TEMPERATURE_K * unit.kelvin,
        enabled_forces=enabled,
        periodic=periodic,
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
    if periodic:
        thermal_context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    mm.LocalEnergyMinimizer.minimize(thermal_context)
    thermal_context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin, 20260810)
    thermal.step(10000)
    start = thermal_context.getState(getPositions=True, getVelocities=True)

    integrator = mm.VerletIntegrator(timestep_fs * unit.femtosecond)
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions(start.getPositions())
    if periodic:
        context.setPeriodicBoxVectors(*start.getPeriodicBoxVectors())
    context.setVelocities(remove_com_velocity(start.getVelocities(), system))
    del thermal_context, thermal

    sample_steps = max(1, round(100_000 / timestep_fs))
    total_steps = round(PRODUCTION_NS * 1_000_000 / timestep_fs)
    times = []
    totals = []
    status = "ok"
    error = None
    started = time.time()
    elapsed = 0
    try:
        while elapsed < total_steps:
            block = min(sample_steps, total_steps - elapsed)
            integrator.step(block)
            elapsed += block
            state = context.getState(getEnergy=True)
            total = (state.getPotentialEnergy() + state.getKineticEnergy()).value_in_unit(
                unit.kilojoule_per_mole
            )
            if not np.isfinite(total):
                raise FloatingPointError("non-finite total energy")
            times.append(elapsed * timestep_fs / 1_000_000)
            totals.append(total)
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    slope = float(np.polyfit(times, totals, 1)[0]) if len(times) > 1 else float("nan")
    slope = slope if np.isfinite(slope) else None
    return {
        "variant": name,
        "periodic": periodic,
        "timestep_fs": timestep_fs,
        "status": status,
        "error": error,
        "elapsed_ns": elapsed * timestep_fs / 1_000_000,
        "wall_seconds": time.time() - started,
        "energy_initial_kj_per_mol": totals[0] if totals else None,
        "energy_final_kj_per_mol": totals[-1] if totals else None,
        "energy_range_kj_per_mol": max(totals) - min(totals) if totals else None,
        "energy_drift_kj_per_mol_per_ns": slope,
        "times_ns": times,
        "total_kj_per_mol": totals,
    }


def main():
    rows = []
    conditions = [("full", VARIANTS["full"], value, True) for value in (1.0, 2.0, 3.0)]
    conditions.append(("full_nonperiodic", VARIANTS["full"], 3.0, False))
    conditions.extend(
        (name, enabled, 3.0, True)
        for name, enabled in VARIANTS.items()
        if name != "full"
    )
    for index, condition in enumerate(conditions, start=1):
        name, enabled, timestep_fs, periodic = condition
        print(f"[{index}/{len(conditions)}] {name} dt={timestep_fs:g} fs periodic={periodic}", flush=True)
        row = run_condition(name, enabled, timestep_fs, periodic)
        rows.append(row)
        write_json(OUTPUT, {"production_ns": PRODUCTION_NS, "runs": rows})
        print(
            f"  status={row['status']} drift={row['energy_drift_kj_per_mol_per_ns']} "
            f"range={row['energy_range_kj_per_mol']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
