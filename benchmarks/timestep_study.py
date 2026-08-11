#!/usr/bin/env python
"""Run resumable NVE/NVT timestep validation experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit

from cranberry.data import data_path
from cranberry.forcefield import BASE_TYPE_TO_CG_NAMES, CranberryForceField, prepare_periodic_positions
from cranberry.md import calculate_langevin_friction


SYSTEMS = {
    "2ntCG": "examples/2ntCG_cg_vs_conect.pdb",
    "1zih": "examples/1zih_cg_vs_conect.pdb",
    "157d": "examples/157d_cg_vs_conect.pdb",
    "1l2x": "examples/1l2x_cg_vs_conect.pdb",
    "rU40": "examples/rU40_cg_vs_conect.pdb",
}
OBSERVABLE_GROUPS = ("bond", "angle", "dihedral", "pucker", "stacking", "pairing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", choices=("nve", "nvt", "ghmc"), required=True)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--timesteps-fs", nargs="+", type=float, required=True)
    parser.add_argument("--temperatures-k", nargs="+", type=float, default=[298.0, 420.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260809])
    parser.add_argument("--equilibration-ps", type=float, default=10.0)
    parser.add_argument("--candidate-burnin-ps", type=float, default=2.0)
    parser.add_argument("--production-ps", type=float, default=20.0)
    parser.add_argument("--sample-ps", type=float, default=0.1)
    parser.add_argument("--equilibration-timestep-fs", type=float, default=1.0)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--precision", default="mixed")
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)
    payload = initialize_output(args)
    completed = {row_key(row) for row in payload["runs"]}
    total = (
        len(args.systems)
        * len(args.temperatures_k)
        * len(args.seeds)
        * len(args.timesteps_fs)
    )
    run_number = len(completed)

    for system_id in args.systems:
        pdb = app.PDBFile(str(data_path(SYSTEMS[system_id])))
        for temperature_k in args.temperatures_k:
            positions = prepare_periodic_positions(pdb.topology, pdb.positions, 4 * unit.nanometer) if args.periodic else pdb.positions
            forcefield = CranberryForceField()
            system = forcefield.createSystem(
                pdb.topology,
                positions=positions,
                temperature=temperature_k * unit.kelvin,
                periodic=args.periodic,
                box_padding=4 * unit.nanometer,
            )
            geometry_spec = geometry_observable_spec(system, pdb.topology, forcefield)
            start = equilibrate(system, pdb, positions, temperature_k, args)
            for seed in args.seeds:
                seeded_start = assign_velocities(start, system, temperature_k, seed, args)
                for timestep_fs in args.timesteps_fs:
                    key = (args.ensemble, system_id, float(temperature_k), seed, float(timestep_fs))
                    if key in completed:
                        continue
                    run_number += 1
                    print(
                        f"[{run_number}/{total}] {args.ensemble} {system_id} "
                        f"T={temperature_k:g}K seed={seed} dt={timestep_fs:g}fs",
                        flush=True,
                    )
                    row = run_condition(
                        system,
                        pdb,
                        seeded_start,
                        args.ensemble,
                        system_id,
                        temperature_k,
                        seed,
                        timestep_fs,
                        geometry_spec,
                        args,
                    )
                    payload["runs"].append(row)
                    write_json_atomic(args.output, payload)
                    completed.add(key)
                    print(
                        f"  status={row['status']} wall={row['wall_seconds']:.2f}s "
                        f"drift={row.get('energy_drift_kbt_per_dof_per_ns')}",
                        flush=True,
                    )
    return 0


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.timesteps_fs
        + args.temperatures_k
        + [args.equilibration_timestep_fs, args.production_ps, args.sample_ps]
    )
    if any(value <= 0 for value in positive):
        raise SystemExit("timesteps, temperatures, production, and sample intervals must be positive")
    if args.equilibration_ps < 0 or args.candidate_burnin_ps < 0:
        raise SystemExit("equilibration and burn-in durations cannot be negative")
    if args.ensemble == "nve" and args.candidate_burnin_ps:
        args.candidate_burnin_ps = 0.0


def initialize_output(args: argparse.Namespace) -> dict:
    if args.output.exists() and not args.overwrite:
        payload = json.loads(args.output.read_text())
        if payload["configuration"] != configuration(args):
            raise SystemExit(f"{args.output} has a different configuration")
        return payload
    payload = {
        "schema_version": 1,
        "study": "cranberry-timestep-validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration(args),
        "environment": {
            "openmm_version": mm.__version__,
            "platform": args.platform,
            "precision": args.precision,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "runs": [],
    }
    write_json_atomic(args.output, payload)
    return payload


def configuration(args: argparse.Namespace) -> dict:
    return {
        "ensemble": args.ensemble,
        "systems": args.systems,
        "timesteps_fs": args.timesteps_fs,
        "temperatures_k": args.temperatures_k,
        "seeds": args.seeds,
        "equilibration_ps": args.equilibration_ps,
        "candidate_burnin_ps": args.candidate_burnin_ps,
        "production_ps": args.production_ps,
        "sample_ps": args.sample_ps,
        "equilibration_timestep_fs": args.equilibration_timestep_fs,
        "periodic": args.periodic,
    }


def row_key(row: dict) -> tuple:
    return (
        row["ensemble"],
        row["system_id"],
        float(row["temperature_k"]),
        int(row["seed"]),
        float(row["timestep_fs"]),
    )


def platform_and_properties(args: argparse.Namespace):
    platform = mm.Platform.getPlatformByName(args.platform)
    properties = {"Precision": args.precision} if args.platform in {"CUDA", "OpenCL"} else {}
    return platform, properties


def equilibrate(system: mm.System, pdb: app.PDBFile, positions, temperature_k: float, args):
    dt = args.equilibration_timestep_fs * unit.femtosecond
    friction = calculate_langevin_friction(pdb.topology, system, temperature_k * unit.kelvin)
    integrator = mm.LangevinMiddleIntegrator(temperature_k * unit.kelvin, friction, dt)
    integrator.setRandomNumberSeed(1)
    platform, properties = platform_and_properties(args)
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions(positions)
    if args.periodic:
        context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    mm.LocalEnergyMinimizer.minimize(context)
    context.setVelocitiesToTemperature(temperature_k * unit.kelvin, 1)
    steps = int(round(args.equilibration_ps * 1000 / args.equilibration_timestep_fs))
    if steps:
        integrator.step(steps)
    state = context.getState(getPositions=True, getVelocities=True)
    result = {
        "positions": state.getPositions(asNumpy=True),
        "box_vectors": state.getPeriodicBoxVectors(asNumpy=True),
    }
    del context, integrator
    return result


def assign_velocities(start: dict, system: mm.System, temperature_k: float, seed: int, args):
    integrator = mm.VerletIntegrator(1 * unit.femtosecond)
    platform, properties = platform_and_properties(args)
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions(start["positions"])
    context.setPeriodicBoxVectors(*start["box_vectors"])
    context.setVelocitiesToTemperature(temperature_k * unit.kelvin, seed)
    state = context.getState(getVelocities=True)
    velocities = remove_com_velocity(state.getVelocities(asNumpy=True), system)
    del context, integrator
    return {**start, "velocities": velocities}


def remove_com_velocity(velocities, system: mm.System):
    values = np.asarray(velocities.value_in_unit(unit.nanometer / unit.picosecond)).copy()
    masses = np.array(
        [system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles())]
    )
    massive = masses > 0
    values[massive] -= np.average(values[massive], axis=0, weights=masses[massive])
    return values * unit.nanometer / unit.picosecond


def run_condition(
    system,
    pdb,
    start,
    ensemble,
    system_id,
    temperature_k,
    seed,
    timestep_fs,
    geometry_spec,
    args,
):
    if ensemble == "nve":
        integrator = mm.VerletIntegrator(timestep_fs * unit.femtosecond)
    elif ensemble == "ghmc":
        from openmmtools.integrators import GHMCIntegrator

        friction = calculate_langevin_friction(
            pdb.topology, system, temperature_k * unit.kelvin
        )
        integrator = GHMCIntegrator(
            temperature_k * unit.kelvin,
            friction,
            timestep_fs * unit.femtosecond,
        )
        integrator.setRandomNumberSeed(seed + int(round(timestep_fs * 1000)))
    else:
        friction = calculate_langevin_friction(
            pdb.topology, system, temperature_k * unit.kelvin
        )
        integrator = mm.LangevinMiddleIntegrator(
            temperature_k * unit.kelvin,
            friction,
            timestep_fs * unit.femtosecond,
        )
        integrator.setRandomNumberSeed(seed + int(round(timestep_fs * 1000)))
    platform, properties = platform_and_properties(args)
    context = mm.Context(system, integrator, platform, properties)
    context.setPeriodicBoxVectors(*start["box_vectors"])
    context.setPositions(start["positions"])
    context.setVelocities(start["velocities"])

    burnin_steps = int(round(args.candidate_burnin_ps * 1000 / timestep_fs))
    production_steps = int(round(args.production_ps * 1000 / timestep_fs))
    sample_steps = max(1, int(round(args.sample_ps * 1000 / timestep_fs)))
    samples = {
        "time_ps": [],
        "potential_kj_per_mol": [],
        "kinetic_kj_per_mol": [],
        "total_kj_per_mol": [],
        "temperature_k": [],
        "radius_of_gyration_nm": [],
        "stiff_bond_lengths_nm": [],
        "p_s3_lengths_nm": [],
        "s2_b1_lengths_nm": [],
        "pucker_phase_degrees": [],
        **{f"{name}_kj_per_mol": [] for name in OBSERVABLE_GROUPS},
    }
    started = time.perf_counter()
    status = "ok"
    error = None
    try:
        if burnin_steps:
            integrator.step(burnin_steps)
        elapsed_steps = 0
        while elapsed_steps < production_steps:
            block = min(sample_steps, production_steps - elapsed_steps)
            integrator.step(block)
            elapsed_steps += block
            record_sample(
                context,
                system,
                samples,
                elapsed_steps * timestep_fs / 1000,
                geometry_spec,
            )
            if not np.isfinite(samples["total_kj_per_mol"][-1]):
                raise FloatingPointError("non-finite total energy")
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    wall_seconds = time.perf_counter() - started
    metrics = summarize(samples, system, temperature_k, ensemble)
    if ensemble == "ghmc":
        ntrials = integrator.getGlobalVariableByName("ntrials")
        naccept = integrator.getGlobalVariableByName("naccept")
        metrics["ghmc_trials"] = int(round(ntrials))
        metrics["ghmc_acceptance"] = float(naccept / ntrials) if ntrials else math.nan
    del context, integrator
    return {
        "ensemble": ensemble,
        "system_id": system_id,
        "temperature_k": temperature_k,
        "seed": seed,
        "timestep_fs": timestep_fs,
        "status": status,
        "error": error,
        "production_ps": args.production_ps,
        "sample_ps": args.sample_ps,
        "geometry_labels": geometry_spec["labels"],
        "wall_seconds": wall_seconds,
        **metrics,
        "samples": samples,
    }


def record_sample(context, system, samples: dict, time_ps: float, geometry_spec: dict) -> None:
    state = context.getState(getEnergy=True, getPositions=True)
    potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    dof = degrees_of_freedom(system, remove_com=True)
    temperature = 2 * kinetic / (dof * unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
        unit.kilojoule_per_mole / unit.kelvin
    ))
    samples["time_ps"].append(time_ps)
    samples["potential_kj_per_mol"].append(potential)
    samples["kinetic_kj_per_mol"].append(kinetic)
    samples["total_kj_per_mol"].append(potential + kinetic)
    samples["temperature_k"].append(temperature)
    positions = state.getPositions(asNumpy=True)
    samples["radius_of_gyration_nm"].append(radius_of_gyration(positions, system))
    geometry = geometry_snapshot(
        positions, state.getPeriodicBoxVectors(asNumpy=True), geometry_spec
    )
    for name, values in geometry.items():
        samples[name].append(values)
    groups = {
        force.getName(): force.getForceGroup()
        for force in system.getForces()
        if force.getName() in OBSERVABLE_GROUPS
    }
    for name in OBSERVABLE_GROUPS:
        if name in groups:
            energy = context.getState(
                getEnergy=True, groups={groups[name]}
            ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        else:
            energy = math.nan
        samples[f"{name}_kj_per_mol"].append(energy)


def degrees_of_freedom(system: mm.System, *, remove_com: bool = False) -> int:
    massive = sum(system.getParticleMass(i) > 0 * unit.dalton for i in range(system.getNumParticles()))
    return 3 * massive - system.getNumConstraints() - (3 if remove_com else 0)


def radius_of_gyration(positions, system: mm.System) -> float:
    xyz = positions.value_in_unit(unit.nanometer)
    masses = np.array(
        [system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles())]
    )
    massive = masses > 0
    center = np.average(xyz[massive], axis=0, weights=masses[massive])
    return float(np.sqrt(np.average(np.sum((xyz[massive] - center) ** 2, axis=1), weights=masses[massive])))


def geometry_observable_spec(
    system: mm.System,
    topology: app.Topology,
    forcefield: CranberryForceField,
) -> dict:
    atoms = list(topology.atoms())
    labels = {
        "stiff_bond_lengths_nm": [],
        "p_s3_lengths_nm": [],
        "s2_b1_lengths_nm": [],
        "pucker_phase_degrees": [],
    }
    stiff_bonds = []
    for force in system.getForces():
        if force.getName() != "bond":
            continue
        for index in range(force.getNumBonds()):
            first, second, _, _ = force.getBondParameters(index)
            first, second = int(first), int(second)
            stiff_bonds.append((first, second))
            labels["stiff_bond_lengths_nm"].append(
                f"{atoms[first].residue.index}:{atoms[first].name}-"
                f"{atoms[second].residue.index}:{atoms[second].name}"
            )

    p_s3 = []
    s2_b1 = []
    pucker_groups = []
    descriptors = [
        name for name in forcefield._params["sugar_sigmoid"] if name != "intercept"
    ]
    weights = np.asarray(
        [float(forcefield._params["sugar_sigmoid"][name]) for name in descriptors]
    )
    intercept = float(forcefield._params["sugar_sigmoid"]["intercept"])
    for residue in topology.residues():
        atom_by_name = {atom.name: atom.index for atom in residue.atoms()}
        base_names = BASE_TYPE_TO_CG_NAMES[residue.name]
        s2_b1.append((atom_by_name["S2"], atom_by_name[base_names[0]]))
        labels["s2_b1_lengths_nm"].append(f"{residue.index}:S2-{base_names[0]}")
        if "P" not in atom_by_name:
            continue
        p_s3.append((atom_by_name["P"], atom_by_name["S3"]))
        labels["p_s3_lengths_nm"].append(f"{residue.index}:P-S3")
        names = ("P", "S3", "S2", *base_names)
        pucker_groups.append(tuple(atom_by_name[name] for name in names))
        labels["pucker_phase_degrees"].append(f"{residue.index}:{residue.name}")
    return {
        "stiff_bonds": stiff_bonds,
        "p_s3": p_s3,
        "s2_b1": s2_b1,
        "pucker_groups": pucker_groups,
        "descriptors": descriptors,
        "weights": weights,
        "intercept": intercept,
        "periodic": system.usesPeriodicBoundaryConditions(),
        "labels": labels,
    }


def geometry_snapshot(positions, box_vectors, spec: dict) -> dict[str, list[float]]:
    xyz = np.asarray(positions.value_in_unit(unit.nanometer))
    if hasattr(box_vectors, "value_in_unit"):
        box = np.asarray(box_vectors.value_in_unit(unit.nanometer))
    else:
        box = np.asarray([vector.value_in_unit(unit.nanometer) for vector in box_vectors])
    inverse_box = np.linalg.inv(box) if spec["periodic"] else None

    def displacement(first, second):
        delta = xyz[second] - xyz[first]
        if inverse_box is not None:
            fractional = delta @ inverse_box
            delta -= np.rint(fractional) @ box
        return delta

    def distance(indices):
        return float(np.linalg.norm(displacement(*indices)))

    def angle(indices):
        first = -displacement(indices[0], indices[1])
        second = displacement(indices[1], indices[2])
        cosine = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
        return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def dihedral(indices):
        b0 = -displacement(indices[0], indices[1])
        b1 = displacement(indices[1], indices[2])
        b2 = displacement(indices[2], indices[3])
        b1 /= np.linalg.norm(b1)
        first = b0 - np.dot(b0, b1) * b1
        second = b2 - np.dot(b2, b1) * b1
        return float(
            np.arctan2(np.dot(np.cross(b1, first), second), np.dot(first, second))
        )

    phases = []
    for group in spec["pucker_groups"]:
        index_by_name = dict(zip(("P", "S3", "S2", "B1", "B2", "B3"), group))
        values = []
        for descriptor in spec["descriptors"]:
            parts = descriptor.split("-")
            transform = None
            if parts[-1] in {"sin", "cos"}:
                transform = parts.pop()
            indices = tuple(index_by_name[name] for name in parts)
            if len(indices) == 2:
                value = distance(indices)
            elif len(indices) == 3:
                value = angle(indices)
            else:
                value = dihedral(indices)
            if transform == "sin":
                value = np.sin(value)
            elif transform == "cos":
                value = np.cos(value)
            values.append(value)
        phases.append(float(np.dot(spec["weights"], values) + spec["intercept"]))
    return {
        "stiff_bond_lengths_nm": [distance(pair) for pair in spec["stiff_bonds"]],
        "p_s3_lengths_nm": [distance(pair) for pair in spec["p_s3"]],
        "s2_b1_lengths_nm": [distance(pair) for pair in spec["s2_b1"]],
        "pucker_phase_degrees": phases,
    }



def summarize(samples: dict, system: mm.System, temperature_k: float, ensemble: str) -> dict:
    result = {"n_samples": len(samples["time_ps"])}
    if not samples["time_ps"]:
        return result
    for name, values in samples.items():
        if name == "time_ps":
            continue
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size:
            result[f"{name}_mean"] = float(np.mean(finite))
            result[f"{name}_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    if ensemble == "nve" and len(samples["time_ps"]) >= 3:
        times = np.asarray(samples["time_ps"])
        energies = np.asarray(samples["total_kj_per_mol"])
        slope, intercept = np.polyfit(times, energies, 1)
        residuals = energies - (slope * times + intercept)
        slope_se = math.sqrt(
            np.sum(residuals**2) / (len(times) - 2) / np.sum((times - np.mean(times)) ** 2)
        )
        thermal_scale = (
            degrees_of_freedom(system, remove_com=True)
            * unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
            * temperature_k
        )
        result.update(
            {
                "energy_drift_kj_per_mol_per_ns": float(slope * 1000),
                "energy_drift_se_kj_per_mol_per_ns": float(slope_se * 1000),
                "energy_drift_kbt_per_dof_per_ns": float(slope * 1000 / thermal_scale),
                "energy_rms_fluctuation_kbt_per_dof": float(
                    np.sqrt(np.mean((energies - np.mean(energies)) ** 2)) / thermal_scale
                ),
            }
        )
    return result


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
