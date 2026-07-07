from __future__ import annotations

import hashlib
import io
import json
import warnings
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TextIO

import openmm as mm
from openmm import LangevinMiddleIntegrator, Platform, unit
from openmm import app

from cranberry.forcefield import FORCE_GROUP_IDS, CranberryForceField, default_model_name, get_model_spec
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


def create_simulation(
    pdb_path: str | Path,
    *,
    model: str = "default",
    temperature=298 * unit.kelvin,
    salt_concentration=150 * unit.millimolar,
    timestep=10 * unit.femtosecond,
    platform: str | None = "CPU",
    restart_from: str | Path | None = None,
) -> app.Simulation:
    """Create an OpenMM Simulation for a canonical CRANBERRY CG PDB."""

    validation = validate_canonical_pdb(pdb_path)
    validation.raise_for_errors()

    temperature = _as_quantity(temperature, unit.kelvin)
    timestep = _as_quantity(timestep, unit.femtosecond)
    salt_concentration = _as_quantity(salt_concentration, unit.millimolar)

    pdb = app.PDBFile(str(pdb_path))
    forcefield = CranberryForceField(model)
    system = forcefield.createSystem(
        pdb.topology,
        positions=pdb.positions,
        temperature=temperature,
        salt_concentration=salt_concentration,
    )
    friction = calculate_langevin_friction(pdb.topology, system, temperature)
    integrator = LangevinMiddleIntegrator(temperature, friction, timestep)

    if platform is None:
        simulation = app.Simulation(pdb.topology, system, integrator)
    else:
        simulation = app.Simulation(pdb.topology, system, integrator, Platform.getPlatformByName(platform))
    simulation.context.setPositions(pdb.positions)
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
    timestep=10 * unit.femtosecond,
    report_interval: int | None = None,
    platform: str | None = "CPU",
    restart_from: str | Path | None = None,
    overwrite: bool = True,
) -> MDRunResult:
    """Run a short OpenMM-native CRANBERRY MD simulation."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if report_interval is None:
        report_interval = max(1, min(1000, steps))
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
    )
    if restart_from_path is not None:
        previous_args = _load_restart_args(result.args_path)
        if previous_args is not None:
            _check_restart_compatibility(previous_args, run_args, result.args_path)

    simulation = create_simulation(
        pdb_path,
        model=model,
        temperature=temperature,
        salt_concentration=salt_concentration,
        timestep=timestep,
        platform=platform,
        restart_from=restart_from_path,
    )
    simulation.reporters.append(app.DCDReporter(str(result.dcd_path), report_interval, append=dcd_append))
    simulation.reporters.append(
        app.StateDataReporter(
            str(result.log_path),
            report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
            totalSteps=steps,
            append=log_append,
        )
    )
    simulation.reporters.append(DetailedEnergyReporter(result.detailed_log_path, report_interval, append=detailed_append))

    _write_args(result.args_path, run_args)
    simulation.step(steps)
    simulation.saveCheckpoint(str(result.checkpoint_path))
    simulation.context.computeVirtualSites()
    state = simulation.context.getState(getPositions=True)
    _write_pdb_with_conect(result.final_pdb_path, simulation.topology, state.getPositions())

    for reporter in simulation.reporters:
        close = getattr(reporter, "close", None)
        if close is not None:
            close()
    return result


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


def _write_pdb_with_conect(path: Path, topology: app.Topology, positions) -> None:
    buffer = io.StringIO()
    app.PDBFile.writeFile(topology, positions, buffer, keepIds=True)
    lines = buffer.getvalue().splitlines()
    if lines and lines[-1].startswith("END"):
        lines = lines[:-1]
    for bond in topology.bonds():
        lines.append(f"CONECT{_pdb_serial(bond.atom1):5d}{_pdb_serial(bond.atom2):5d}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _pdb_serial(atom) -> int:
    try:
        serial = int(atom.id)
    except (TypeError, ValueError):
        return atom.index + 1
    return serial if 0 < serial <= 99999 else atom.index + 1


def _present_force_group_names(system: mm.System) -> list[str]:
    names_by_group = {system.getForce(i).getForceGroup(): system.getForce(i).getName() for i in range(system.getNumForces())}
    return [name for name, group in FORCE_GROUP_IDS.items() if group in names_by_group and names_by_group[group] == name]


def _build_args(
    *,
    pdb_path,
    model,
    steps,
    report_interval,
    temperature,
    salt_concentration,
    timestep,
    platform,
    restart_from,
    append_outputs,
    dcd_append,
    log_append,
    detailed_append,
    overwrite,
) -> dict[str, object]:
    temperature = _as_quantity(temperature, unit.kelvin)
    salt_concentration = _as_quantity(salt_concentration, unit.millimolar)
    timestep = _as_quantity(timestep, unit.femtosecond)
    pdb_path = Path(pdb_path)
    return {
        "schema_version": 1,
        "run_kind": "md",
        "pdb_path": str(pdb_path),
        "pdb_sha256": _sha256(pdb_path),
        "model": default_model_name() if model == "default" else model,
        "steps": steps,
        "report_interval": report_interval,
        "temperature_kelvin": temperature.value_in_unit(unit.kelvin),
        "salt_millimolar": salt_concentration.value_in_unit(unit.millimolar),
        "timestep_femtosecond": timestep.value_in_unit(unit.femtosecond),
        "platform": platform,
        "openmm_version": getattr(mm, "__version__", mm.version.version),
        "cranberry_version": _cranberry_version(),
        "restart_from": str(restart_from) if restart_from is not None else None,
        "append_outputs": append_outputs,
        "dcd_append": dcd_append,
        "log_append": log_append,
        "detailed_append": detailed_append,
        "overwrite": overwrite,
    }


def _write_args(path: Path, args: dict[str, object]) -> None:
    _archive_args_if_distinct(path, args)
    path.write_text(json.dumps(args, indent=2, sort_keys=True) + "\n")


def _archive_args_if_distinct(path: Path, args: dict[str, object]) -> Path | None:
    if not path.exists():
        return None
    try:
        existing = json.loads(path.read_text())
    except json.JSONDecodeError:
        archive_path = _next_args_history_path(path)
        archive_path.write_text(path.read_text())
        return archive_path
    if _canonical_json(existing) == _canonical_json(args):
        return None
    archive_path = _next_args_history_path(path)
    archive_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    return archive_path


def _next_args_history_path(args_path: Path) -> Path:
    history_dir = args_path.parent / "args_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = history_dir / f"{index:06d}_args.json"
        if not candidate.exists():
            return candidate
        index += 1


def _load_restart_args(args_path: Path) -> dict[str, object] | None:
    if not args_path.exists():
        warnings.warn(
            f"Restart compatibility cannot be checked because {args_path} does not exist.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    try:
        return json.loads(args_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot validate restart compatibility because {args_path} is not valid JSON") from exc


def _check_restart_compatibility(previous: dict[str, object], current: dict[str, object], args_path: Path) -> None:
    errors = []
    for key in (
        "run_kind",
        "model",
        "pdb_sha256",
        "temperature_kelvin",
        "salt_millimolar",
        "timestep_femtosecond",
    ):
        if previous.get(key) != current.get(key):
            errors.append(f"{key}: previous={previous.get(key)!r}, current={current.get(key)!r}")
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Restart arguments are incompatible with {args_path}:\n{details}")
    for key in ("platform", "openmm_version", "cranberry_version"):
        if previous.get(key) != current.get(key):
            warnings.warn(
                f"Restart {key} differs from {args_path}: previous={previous.get(key)!r}, current={current.get(key)!r}",
                RuntimeWarning,
                stacklevel=3,
            )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cranberry_version() -> str:
    try:
        return version("cranberry-rna")
    except PackageNotFoundError:  # pragma: no cover - editable source tree before install
        return "0+unknown"


def _as_quantity(value, target_unit):
    if hasattr(value, "unit"):
        return value.in_units_of(target_unit)
    return value * target_unit
