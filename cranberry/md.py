from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TextIO

try:
    __version__ = version("cranberry-rna")
except PackageNotFoundError:  # pragma: no cover - editable source tree before install
    __version__ = "0+unknown"

import openmm as mm
from openmm import LangevinMiddleIntegrator, Platform, unit
from openmm import app

from cranberry.forcefield import FORCE_GROUP_IDS, CranberryForceField, default_model_name, get_model_spec, prepare_periodic_positions, validate_periodic_box_cutoffs
from cranberry.pdbio import write_pdb_with_conect
from cranberry.validation import validate_canonical_pdb


@dataclass(frozen=True)
class MDRunResult:
    output_dir: Path
    dcd_path: Path
    log_path: Path
    detailed_log_path: Path
    args_path: Path
    checkpoint_path: Path
    final_pdb_path: Path
    restart_from_path: Path | None
    steps: int
    report_interval: int
    n_record: int
    minimization_report_path: Path | None
    actual_platform: str | None


def create_simulation(
    pdb_path: str | Path,
    *,
    model: str = "default",
    temperature=298 * unit.kelvin,
    salt_concentration=150 * unit.millimolar,
    timestep=5 * unit.femtosecond,
    platform: str | None = "CPU",
    restart_from: str | Path | None = None,
    periodic: bool = False,
    box_padding=2.0 * unit.nanometer,
) -> app.Simulation:
    """Create an OpenMM Simulation for a canonical CRANBERRY CG PDB."""

    validation = validate_canonical_pdb(pdb_path)
    validation.raise_for_errors()

    temperature = _as_quantity(temperature, unit.kelvin)
    timestep = _as_quantity(timestep, unit.femtosecond)
    salt_concentration = _as_quantity(salt_concentration, unit.millimolar)

    pdb = app.PDBFile(str(pdb_path))
    positions = prepare_periodic_positions(pdb.topology, pdb.positions, box_padding) if periodic else pdb.positions
    forcefield = CranberryForceField(model)
    system = forcefield.createSystem(
        pdb.topology,
        positions=positions,
        temperature=temperature,
        salt_concentration=salt_concentration,
        periodic=periodic,
        box_padding=box_padding,
    )
    validate_periodic_box_cutoffs(system)
    friction = calculate_langevin_friction(pdb.topology, system, temperature)
    integrator = LangevinMiddleIntegrator(temperature, friction, timestep)

    if platform is None:
        simulation = app.Simulation(pdb.topology, system, integrator)
    else:
        simulation = app.Simulation(pdb.topology, system, integrator, Platform.getPlatformByName(platform))
    simulation.context.setPositions(positions)
    if restart_from is None:
        simulation.context.setVelocitiesToTemperature(temperature)
    else:
        restart_from = Path(restart_from)
        if not restart_from.exists():
            raise FileNotFoundError(f"Checkpoint not found: {restart_from}")
        simulation.loadCheckpoint(str(restart_from))
    return simulation


def run_md(
    pdb_path: str | Path,
    *,
    steps: int,
    output_dir: str | Path = ".",
    model: str = "default",
    temperature=298 * unit.kelvin,
    salt_concentration=150 * unit.millimolar,
    timestep=5 * unit.femtosecond,
    report_interval: int | None = None,
    n_record: int = 1000,
    platform: str | None = "CPU",
    restart_from: str | Path | None = None,
    overwrite: bool = True,
    write_minimization_report: bool = False,
    periodic: bool = False,
    box_padding=2.0 * unit.nanometer,
    enforce_periodic_output: bool = False,
    log_progress: bool = False,
) -> MDRunResult:
    """Run a short OpenMM-native CRANBERRY MD simulation."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if n_record < 1:
        raise ValueError("n_record must be at least 1")
    if report_interval is None:
        report_interval = max(1, int(steps / n_record))
    if report_interval < 1:
        raise ValueError("report_interval must be at least 1")

    output_dir = Path(output_dir)
    restart_from_path = Path(restart_from) if restart_from is not None else None
    if restart_from_path is not None and not restart_from_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {restart_from_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = MDRunResult(
        output_dir=output_dir,
        dcd_path=output_dir / "output.dcd",
        log_path=output_dir / "log",
        detailed_log_path=output_dir / "detailed.log",
        args_path=output_dir / "args.json",
        checkpoint_path=output_dir / "checkpoint.chk",
        final_pdb_path=output_dir / "final.pdb",
        restart_from_path=restart_from_path,
        steps=steps,
        report_interval=report_interval,
        n_record=n_record,
        minimization_report_path=(output_dir / "minimization_report.json") if write_minimization_report else None,
        actual_platform=None,
    )
    append_outputs = restart_from_path is not None
    dcd_append = append_outputs and result.dcd_path.exists()
    log_append = append_outputs and result.log_path.exists()
    detailed_append = append_outputs and result.detailed_log_path.exists()
    if append_outputs:
        missing = [
            str(path)
            for path, should_append in (
                (result.dcd_path, dcd_append),
                (result.log_path, log_append),
                (result.detailed_log_path, detailed_append),
            )
            if not should_append
        ]
        if missing:
            warnings.warn(
                "Restart output files are missing and will be created starting from the checkpoint step: "
                + ", ".join(missing),
                RuntimeWarning,
                stacklevel=2,
            )
    elif not overwrite:
        paths = [result.dcd_path, result.log_path, result.detailed_log_path, result.args_path, result.checkpoint_path, result.final_pdb_path]
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing MD outputs: " + ", ".join(existing))

    model_name = get_model_spec(model).name
    run_args = _build_args(
        pdb_path=pdb_path,
        model=model_name,
        steps=steps,
        report_interval=report_interval,
        n_record=n_record,
        temperature=temperature,
        salt_concentration=salt_concentration,
        timestep=timestep,
        platform=platform,
        restart_from=restart_from_path,
        append_outputs=append_outputs,
        dcd_append=dcd_append,
        log_append=log_append,
        detailed_append=detailed_append,
        overwrite=overwrite,
        write_minimization_report=write_minimization_report,
        periodic=periodic,
        box_padding=box_padding,
        enforce_periodic_output=enforce_periodic_output,
        log_progress=log_progress,
        actual_platform=None,
    )
    if restart_from_path is not None:
        previous_args = _load_restart_args(result.args_path)
        if previous_args is not None:
            _check_restart_compatibility(previous_args, run_args, result.args_path)
        else:
            warnings.warn(
                "Restart compatibility cannot be checked without args.json from a previous run; proceeding with checkpoint state only.",
                RuntimeWarning,
                stacklevel=2,
            )

    simulation = create_simulation(
        pdb_path,
        model=model,
        temperature=temperature,
        salt_concentration=salt_concentration,
        timestep=timestep,
        platform=platform,
        restart_from=restart_from_path,
        periodic=periodic,
        box_padding=box_padding,
    )
    actual_platform = simulation.context.getPlatform().getName()
    _warn_platform_mismatch(platform, actual_platform)
    run_args["actual_platform"] = actual_platform
    result = MDRunResult(
        output_dir=result.output_dir,
        dcd_path=result.dcd_path,
        log_path=result.log_path,
        detailed_log_path=result.detailed_log_path,
        args_path=result.args_path,
        checkpoint_path=result.checkpoint_path,
        final_pdb_path=result.final_pdb_path,
        restart_from_path=result.restart_from_path,
        steps=result.steps,
        report_interval=result.report_interval,
        n_record=result.n_record,
        minimization_report_path=result.minimization_report_path,
        actual_platform=actual_platform,
    )
    _write_args(result.args_path, run_args)
    simulation.reporters.append(app.DCDReporter(str(result.dcd_path), report_interval, append=dcd_append, enforcePeriodicBox=enforce_periodic_output))
    simulation.reporters.append(
        app.StateDataReporter(
            str(result.log_path),
            report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            progress=log_progress,
            elapsedTime=True,
            speed=True,
            remainingTime=True,
            totalSteps=steps,
            append=log_append,
        )
    )
    simulation.reporters.append(DetailedEnergyReporter(result.detailed_log_path, report_interval, append=detailed_append))
    simulation.reporters.append(app.CheckpointReporter(str(result.checkpoint_path), report_interval))

    minimization_summary = _minimize_and_report(simulation, result.minimization_report_path)
    print(
        "minimization: "
        f"potential_energy_before={minimization_summary['before_kj_per_mol']:.8f} kJ/mol, "
        f"after={minimization_summary['after_kj_per_mol']:.8f} kJ/mol"
    )

    simulation.step(steps)
    simulation.saveCheckpoint(str(result.checkpoint_path))
    simulation.context.computeVirtualSites()
    state = simulation.context.getState(getPositions=True)
    write_pdb_with_conect(result.final_pdb_path, simulation.topology, state.getPositions())

    for reporter in simulation.reporters:
        close = getattr(reporter, "close", None)
        if close is not None:
            close()
    return result



def _warn_platform_mismatch(requested_platform: str | None, actual_platform: str | None) -> None:
    if requested_platform is None or actual_platform is None:
        return
    if requested_platform != actual_platform:
        warnings.warn(
            f"Requested OpenMM platform {requested_platform!r}, but actual platform is {actual_platform!r}.",
            RuntimeWarning,
            stacklevel=2,
        )


def _minimize_and_report(simulation: app.Simulation, report_path: Path | None) -> dict[str, object]:
    before = _energy_snapshot(simulation)
    simulation.minimizeEnergy()
    after = _energy_snapshot(simulation)
    summary = {
        "before_kj_per_mol": before["potential_energy_kj_per_mol"],
        "after_kj_per_mol": after["potential_energy_kj_per_mol"],
        "force_groups_before_kj_per_mol": before["force_groups_kj_per_mol"],
        "force_groups_after_kj_per_mol": after["force_groups_kj_per_mol"],
    }
    if report_path is not None:
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _energy_snapshot(simulation: app.Simulation) -> dict[str, object]:
    state = simulation.context.getState(getEnergy=True)
    force_groups = {}
    for name in _present_force_group_names(simulation.system):
        group = FORCE_GROUP_IDS[name]
        group_state = simulation.context.getState(getEnergy=True, groups={group})
        force_groups[name] = group_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return {
        "potential_energy_kj_per_mol": state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        "force_groups_kj_per_mol": force_groups,
    }

def _as_quantity(value, default_unit):
    if hasattr(value, "unit"):
        return value
    return value * default_unit


def _build_args(*, pdb_path, model, steps, report_interval, n_record, temperature, salt_concentration, timestep, platform, restart_from, append_outputs, dcd_append, log_append, detailed_append, overwrite, write_minimization_report, periodic, box_padding, enforce_periodic_output, log_progress, actual_platform):
    pdb_path = Path(pdb_path)
    return {
        "schema_version": 1,
        "run_kind": "md",
        "pdb_path": str(pdb_path),
        "pdb_sha256": _sha256(pdb_path),
        "model": model,
        "steps": int(steps),
        "report_interval": int(report_interval),
        "n_record": int(n_record),
        "temperature_kelvin": float(_as_quantity(temperature, unit.kelvin).value_in_unit(unit.kelvin)),
        "salt_millimolar": float(_as_quantity(salt_concentration, unit.millimolar).value_in_unit(unit.millimolar)),
        "timestep_femtosecond": float(_as_quantity(timestep, unit.femtosecond).value_in_unit(unit.femtosecond)),
        "platform": platform,
        "actual_platform": actual_platform,
        "restart_from": str(restart_from) if restart_from is not None else None,
        "append_outputs": bool(append_outputs),
        "dcd_append": bool(dcd_append),
        "log_append": bool(log_append),
        "detailed_append": bool(detailed_append),
        "overwrite": bool(overwrite),
        "write_minimization_report": bool(write_minimization_report),
        "periodic": bool(periodic),
        "box_padding_nanometer": float(_as_quantity(box_padding, unit.nanometer).value_in_unit(unit.nanometer)),
        "enforce_periodic_output": bool(enforce_periodic_output),
        "log_progress": bool(log_progress),
        "cranberry_version": __version__,
        "openmm_version": getattr(mm, "__version__", None),
    }


def _write_args(path: Path, args: dict) -> None:
    path = Path(path)
    payload = json.dumps(args, indent=2, sort_keys=True)
    text = payload + "\n"
    if path.exists():
        existing = path.read_text()
        if existing == text:
            return
        history_path = _next_args_history_path(path.parent / "args_history")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(existing if existing.endswith("\n") else existing + "\n")
    path.write_text(text)


def _load_restart_args(path: Path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"args.json at {path} is not valid JSON: {exc}") from exc


def _check_restart_compatibility(previous_args: dict, run_args: dict, args_path: Path) -> None:
    error_fields = ["run_kind", "pdb_sha256", "model", "temperature_kelvin", "salt_millimolar", "timestep_femtosecond", "periodic", "box_padding_nanometer", "enforce_periodic_output", "log_progress"]
    for field in error_fields:
        if previous_args.get(field) != run_args.get(field):
            raise ValueError(f"Restart compatibility check failed for {field} in {args_path}")

    warning_fields = ["platform", "cranberry_version", "openmm_version"]
    for field in warning_fields:
        if previous_args.get(field) != run_args.get(field):
            warnings.warn(
                f"Restart metadata differs for {field}; continuing anyway.",
                RuntimeWarning,
                stacklevel=2,
            )


def _next_args_history_path(history_dir: Path) -> Path:
    history_dir = Path(history_dir)
    index = 1
    while True:
        candidate = history_dir / f"{index:06d}_args.json"
        if not candidate.exists():
            return candidate
        index += 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _present_force_group_names(system: mm.System) -> list[str]:
    names_by_group = {system.getForce(i).getForceGroup(): system.getForce(i).getName() for i in range(system.getNumForces())}
    return [name for name, group in FORCE_GROUP_IDS.items() if names_by_group.get(group) == name]


def calculate_langevin_friction(topology: app.Topology, system: mm.System, temperature) -> unit.Quantity:
    """Return the legacy length-dependent Langevin friction coefficient."""

    temperature = _as_quantity(temperature, unit.kelvin)
    n_residues = sum(1 for _ in topology.residues())
    if n_residues < 1:
        raise ValueError("topology must contain at least one residue")
    diffusion = 4.58e-10 * unit.meter**2 / unit.second * n_residues ** (-0.39)
    total_mass = sum((system.getParticleMass(i) for i in range(system.getNumParticles())), 0 * unit.dalton)
    total_mass = total_mass.in_units_of(unit.gram / unit.mole)
    gamma = unit.MOLAR_GAS_CONSTANT_R * temperature / (diffusion * total_mass)
    return gamma.in_units_of(unit.picosecond**-1)


class DetailedEnergyReporter:
    """OpenMM reporter for total and named force-group potential energies."""

    def __init__(self, file: str | Path | TextIO, reportInterval: int, append: bool = False):
        if reportInterval < 1:
            raise ValueError("reportInterval must be at least 1")
        self._reportInterval = reportInterval
        self._has_initialized = append
        self._opened_file = isinstance(file, (str, Path))
        self._out = open(file, "a" if append else "w") if self._opened_file else file

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return {"steps": steps, "periodic": None, "include": ["energy"]}

    def report(self, simulation, state):
        names = _present_force_group_names(simulation.system)
        if not self._has_initialized:
            header = ["Step", "Time (ps)", "Potential Energy (kJ/mole)"]
            header.extend(f"{name} (kJ/mole)" for name in names)
            self._out.write("#" + ",".join(json.dumps(item) for item in header) + "\n")
            self._has_initialized = True
        values = [
            str(simulation.currentStep),
            f"{state.getTime().value_in_unit(unit.picosecond):.8f}",
            f"{state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):.8f}",
        ]
        for name in names:
            group = FORCE_GROUP_IDS[name]
            energy = simulation.context.getState(getEnergy=True, groups={group}).getPotentialEnergy()
            values.append(f"{energy.value_in_unit(unit.kilojoule_per_mole):.8f}")
        self._out.write(",".join(values) + "\n")
        self._out.flush()

    def close(self) -> None:
        if self._opened_file and not self._out.closed:
            self._out.close()

    def __del__(self):  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

