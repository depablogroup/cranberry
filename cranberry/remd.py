from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

import numpy as np
import openmm as mm
from openmm import app, unit

from cranberry.forcefield import (
    CranberryForceField,
    get_model_spec,
    prepare_common_periodic_positions,
    validate_periodic_box_cutoffs,
)
from cranberry.md import _present_force_group_names, _sha256, _write_args, calculate_langevin_friction
from cranberry.validation import validate_canonical_pdb

try:
    __version__ = version('cranberry-rna')
except PackageNotFoundError:  # pragma: no cover - editable source tree before install
    __version__ = '0+unknown'

DEFAULT_REMD_T_MIN_K = 298.0
DEFAULT_REMD_T_MAX_K = 600.0
DEFAULT_REMD_N_REPLICAS = 8
DEFAULT_REMD_SWAP_STEPS = 5000
DEFAULT_REMD_N_RECORD = 1000
DEFAULT_REMD_N_ANALYSIS = 0
DEFAULT_REMD_MCMC_MOVE = 'LangevinDynamicsMove'


@dataclass(frozen=True)
class TemperatureLadderSpec:
    """Describe the temperature schedule used by `cranberry remd`."""

    temperatures: tuple[float, ...] | None = None
    t_min: float = DEFAULT_REMD_T_MIN_K
    t_max: float = DEFAULT_REMD_T_MAX_K
    n_replicas: int = DEFAULT_REMD_N_REPLICAS

    def resolve(self) -> tuple[float, ...] | None:
        """Return an explicit ladder if one was provided."""

        if self.temperatures is None:
            return None
        resolved = tuple(float(temperature) for temperature in self.temperatures)
        if not resolved:
            raise ValueError('temperatures must not be empty')
        if any(temperature <= 0 for temperature in resolved):
            raise ValueError('temperatures must be positive')
        if any(b <= a for a, b in zip(resolved, resolved[1:])):
            raise ValueError('temperatures must be strictly increasing')
        return resolved

    def validate_defaults(self) -> None:
        if self.temperatures is not None:
            self.resolve()
            return
        if self.n_replicas < 1:
            raise ValueError('n_replicas must be at least 1')
        if self.t_min <= 0 or self.t_max <= 0:
            raise ValueError('t_min and t_max must be positive')
        if self.t_max < self.t_min:
            raise ValueError('t_max must be greater than or equal to t_min')


@dataclass(frozen=True)
class RemdRunConfig:
    """Configuration for a `cranberry remd` run."""

    pdb_path: Path
    steps: int
    temperature_ladder: TemperatureLadderSpec = field(default_factory=TemperatureLadderSpec)
    output_dir: Path = Path('.')
    model: str = 'default'
    swap_steps: int = DEFAULT_REMD_SWAP_STEPS
    n_record: int = DEFAULT_REMD_N_RECORD
    n_analysis: int = DEFAULT_REMD_N_ANALYSIS
    salt_concentration_millimolar: float = 150.0
    timestep_femtosecond: float = 5.0
    platform: str | None = 'CPU'
    restart_from: Path | None = None
    extra_start_pdb: Path | None = None
    overwrite: bool = False
    write_dcd: bool = False
    dcd_mode: str = 'replica'
    periodic: bool = False
    box_padding_nanometer: float = 2.0


@dataclass(frozen=True)
class RemdRunResult:
    """Summary of the REMD output layout."""

    output_dir: Path
    output_netcdf_path: Path
    args_path: Path
    output_dcd_path: Path | tuple[Path, ...] | None
    output_dcd_manifest_path: Path | None
    restart_from_path: Path | None
    temperatures_kelvin: tuple[float, ...]
    steps: int
    iterations: int
    swap_steps: int
    checkpoint_interval: int
    online_analysis_interval: int | None
    actual_platform: str | None
    jax_platform_name_env: str | None


@dataclass(frozen=True)
class _MpiRuntime:
    enabled: bool = False
    rank: int = 0
    size: int = 1


def run_remd(config: RemdRunConfig) -> RemdRunResult:
    """Run replica exchange MD through the optional OpenMMTools stack."""

    validation = validate_canonical_pdb(config.pdb_path)
    validation.raise_for_errors()
    if config.extra_start_pdb is not None:
        extra_validation = validate_canonical_pdb(config.extra_start_pdb)
        extra_validation.raise_for_errors()
    if config.steps < 1:
        raise ValueError('steps must be at least 1')
    if config.swap_steps < 1:
        raise ValueError('swap_steps must be at least 1')
    if config.n_record < 1:
        raise ValueError('n_record must be at least 1')
    if config.n_analysis < 0:
        raise ValueError('n_analysis must be zero or greater')

    n_iterations = max(1, int(config.steps / config.swap_steps))
    checkpoint_interval = max(1, int(config.steps / (config.swap_steps * config.n_record)))
    online_analysis_interval = None if config.n_analysis == 0 else max(1, int(n_iterations / config.n_analysis))
    jax_platform_name_env = os.environ.get('JAX_PLATFORM_NAME')

    ladder_spec = config.temperature_ladder
    ladder_spec.validate_defaults()
    explicit_temperatures = ladder_spec.resolve()

    output_dir = Path(config.output_dir) if config.restart_from is None else Path(config.restart_from).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_netcdf_path = Path(config.restart_from) if config.restart_from is not None else output_dir / 'output.nc'
    args_path = output_dir / 'args.json'

    expected_outputs = [args_path, output_netcdf_path]
    if config.write_dcd:
        expected_outputs.extend(_expected_dcd_paths(output_dir, config.dcd_mode, ladder_spec))
        if config.dcd_mode == 'temperature':
            expected_outputs.append(output_dir / 'output_temperature_labels.txt')
    if config.restart_from is None and not config.overwrite:
        _ensure_outputs_available(*expected_outputs)
    elif config.restart_from is not None and not output_netcdf_path.exists():
        raise FileNotFoundError(f'REMD NetCDF restart not found: {output_netcdf_path}')
    if config.restart_from is not None:
        _validate_restart_metadata(args_path, config, get_model_spec(config.model).name)

    pdb = app.PDBFile(str(config.pdb_path))
    extra_pdb = _load_extra_start_pdb(pdb, config.extra_start_pdb)
    raw_position_sets = [pdb.positions]
    if extra_pdb is not None:
        raw_position_sets.append(extra_pdb.positions)
    box_padding = config.box_padding_nanometer * unit.nanometer
    if config.periodic:
        start_positions = prepare_common_periodic_positions(
            pdb.topology,
            raw_position_sets,
            box_padding,
        )
    else:
        start_positions = tuple(raw_position_sets)
    positions = start_positions[0]
    common_box_vectors = pdb.topology.getPeriodicBoxVectors() if config.periodic else None
    forcefield = CranberryForceField(config.model)
    reference_temperature = explicit_temperatures[0] if explicit_temperatures is not None else ladder_spec.t_min
    system = forcefield.createSystem(
        pdb.topology,
        positions=positions,
        temperature=reference_temperature * unit.kelvin,
        salt_concentration=config.salt_concentration_millimolar * unit.millimolar,
        periodic=config.periodic,
        box_padding=box_padding,
    )
    if common_box_vectors is not None:
        pdb.topology.setPeriodicBoxVectors(common_box_vectors)
        system.setDefaultPeriodicBoxVectors(*common_box_vectors)
    validate_periodic_box_cutoffs(system)
    openmmtools, mcmc, states, multistate = _load_openmmtools()
    mpiplus = _load_mpiplus()
    mpi_runtime = _detect_mpi_runtime(mpiplus)
    move_temperatures = _expected_temperatures(ladder_spec)
    collision_rates = tuple(
        calculate_langevin_friction(pdb.topology, system, temperature * unit.kelvin)
        for temperature in move_temperatures
    )
    move = [
        mcmc.LangevinDynamicsMove(
            timestep=config.timestep_femtosecond * unit.femtosecond,
            collision_rate=collision_rate,
            n_steps=config.swap_steps,
            reassign_velocities=True,
        )
        for collision_rate in collision_rates
    ]
    sampler_cls = multistate.ParallelTemperingSampler
    actual_platform = None

    if config.restart_from is None:
        reporter = multistate.MultiStateReporter(str(output_netcdf_path), checkpoint_interval=checkpoint_interval)
        sampler_kwargs = {'mcmc_moves': move, 'number_of_iterations': n_iterations}
        if online_analysis_interval is not None:
            sampler_kwargs['online_analysis_interval'] = online_analysis_interval
        sampler = sampler_cls(**sampler_kwargs)
        _apply_sampler_platform(sampler, openmmtools, config.platform)
        reference_state = states.ThermodynamicState(system, temperature=reference_temperature * unit.kelvin)
        n_replicas = len(explicit_temperatures) if explicit_temperatures is not None else ladder_spec.n_replicas
        sampler_box_vectors = pdb.topology.getPeriodicBoxVectors() if config.periodic else None
        sampler_states = [
            states.SamplerState(start_positions[index % len(start_positions)], box_vectors=sampler_box_vectors)
            for index in range(n_replicas)
        ]
        create_kwargs = _build_parallel_tempering_kwargs(ladder_spec, explicit_temperatures)
        sampler.create(reference_state, sampler_states, reporter, **create_kwargs)
        actual_platform = _actual_sampler_platform(sampler, reference_state)
        _warn_platform_mismatch(config.platform, actual_platform)
        _warn_cuda_online_analysis(actual_platform, online_analysis_interval)
        if _is_mpi_root(mpi_runtime):
            _write_args(
                args_path,
                _build_remd_args(
                    config=config,
                    model_name=get_model_spec(config.model).name,
                    output_netcdf_path=output_netcdf_path,
                    output_dcd_path=None,
                    output_dcd_manifest_path=None,
                    temperatures_kelvin=_expected_temperatures(ladder_spec),
                    n_iterations=n_iterations,
                    checkpoint_interval=checkpoint_interval,
                    online_analysis_interval=online_analysis_interval,
                    actual_platform=actual_platform,
                    mcmc_move=DEFAULT_REMD_MCMC_MOVE,
                    collision_rates=collision_rates,
                    openmmtools_version=getattr(openmmtools, '__version__', None),
                    pdb=pdb,
                    system=system,
                ),
            )
        sampler.minimize()
        sampler.run()
        stored_temperatures = _read_temperatures_from_reporter(reporter) if _is_mpi_root(mpi_runtime) else None
        stored_temperatures = _mpi_bcast(mpiplus, stored_temperatures)
        _close_if_present(reporter)
    else:
        if _is_mpi_root(mpi_runtime):
            reporter = multistate.MultiStateReporter(str(output_netcdf_path), open_mode='r', checkpoint_interval=checkpoint_interval)
            try:
                stored_temperatures = _read_temperatures_from_reporter(reporter)
                expected_temperatures = _expected_temperatures(ladder_spec)
                if stored_temperatures != expected_temperatures:
                    raise ValueError(
                        'REMD restart ladder does not match the existing NetCDF storage: '
                        f'expected {expected_temperatures}, found {stored_temperatures}'
                    )
            finally:
                _close_if_present(reporter)
        else:
            stored_temperatures = None
        stored_temperatures = _mpi_bcast(mpiplus, stored_temperatures)
        sampler = _sampler_from_storage(sampler_cls, output_netcdf_path, mpiplus, mpi_runtime)
        _apply_sampler_platform(sampler, openmmtools, config.platform)
        reference_state = states.ThermodynamicState(system, temperature=stored_temperatures[0] * unit.kelvin)
        actual_platform = _actual_sampler_platform(sampler, reference_state)
        _warn_platform_mismatch(config.platform, actual_platform)
        _warn_cuda_online_analysis(actual_platform, online_analysis_interval)
        if _is_mpi_root(mpi_runtime):
            _write_args(
                args_path,
                _build_remd_args(
                    config=config,
                    model_name=get_model_spec(config.model).name,
                    output_netcdf_path=output_netcdf_path,
                    output_dcd_path=None,
                    output_dcd_manifest_path=None,
                    temperatures_kelvin=stored_temperatures,
                    n_iterations=n_iterations,
                    checkpoint_interval=checkpoint_interval,
                    online_analysis_interval=online_analysis_interval,
                    actual_platform=actual_platform,
                    mcmc_move=DEFAULT_REMD_MCMC_MOVE,
                    collision_rates=collision_rates,
                    openmmtools_version=getattr(openmmtools, '__version__', None),
                    pdb=pdb,
                    system=system,
                ),
            )
        sampler.extend(n_iterations)

    output_dcd_path = None
    output_dcd_manifest_path = None
    if config.write_dcd and _is_mpi_root(mpi_runtime):
        output_dcd_path = translate_netcdf_to_dcd(
            output_netcdf_path,
            pdb_path=config.pdb_path,
            output_dir=output_dir,
            output_mode=config.dcd_mode,
            overwrite=config.overwrite,
        )
        if config.dcd_mode == 'temperature':
            output_dcd_manifest_path = output_dir / 'output_temperature_labels.txt'

    if (output_dcd_path is not None or output_dcd_manifest_path is not None) and _is_mpi_root(mpi_runtime):
        _write_args(
            args_path,
            _build_remd_args(
                config=config,
                model_name=get_model_spec(config.model).name,
                output_netcdf_path=output_netcdf_path,
                output_dcd_path=output_dcd_path,
                output_dcd_manifest_path=output_dcd_manifest_path,
                temperatures_kelvin=stored_temperatures,
                n_iterations=n_iterations,
                checkpoint_interval=checkpoint_interval,
                online_analysis_interval=online_analysis_interval,
                actual_platform=actual_platform,
                mcmc_move=DEFAULT_REMD_MCMC_MOVE,
                collision_rates=collision_rates,
                openmmtools_version=getattr(openmmtools, '__version__', None),
                pdb=pdb,
                system=system,
            ),
        )

    _close_if_present(sampler)

    return RemdRunResult(
        output_dir=output_dir,
        output_netcdf_path=output_netcdf_path,
        args_path=args_path,
        output_dcd_path=output_dcd_path,
        output_dcd_manifest_path=output_dcd_manifest_path,
        restart_from_path=Path(config.restart_from) if config.restart_from is not None else None,
        temperatures_kelvin=stored_temperatures,
        steps=config.steps,
        iterations=n_iterations,
        swap_steps=config.swap_steps,
        checkpoint_interval=checkpoint_interval,
        online_analysis_interval=online_analysis_interval,
        actual_platform=actual_platform,
        jax_platform_name_env=jax_platform_name_env,
    )


def translate_netcdf_to_dcd(
    netcdf_path: str | Path,
    *,
    pdb_path: str | Path,
    output_dir: str | Path | None = None,
    output_mode: str = 'replica',
    overwrite: bool = False,
) -> Path | tuple[Path, ...] | None:
    """Translate a REMD NetCDF trajectory to DCD files."""

    netcdf_path = Path(netcdf_path)
    pdb_path = Path(pdb_path)
    output_dir = Path(output_dir) if output_dir is not None else netcdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    mpiplus = _load_mpiplus()
    mpi_runtime = _detect_mpi_runtime(mpiplus)
    if not _is_mpi_root(mpi_runtime):
        return None

    _, _, _, multistate = _load_openmmtools()
    reporter = multistate.MultiStateReporter(str(netcdf_path), open_mode='r', checkpoint_interval=1)
    pdb = app.PDBFile(str(pdb_path))
    iterations = list(reporter.read_checkpoint_iterations())
    replica_thermodynamic_states = np.asarray(reporter.read_replica_thermodynamic_states())
    if replica_thermodynamic_states.size == 0:
        raise ValueError('REMD NetCDF file does not contain replica state assignments')
    thermodynamic_states, _ = reporter.read_thermodynamic_states()
    temperatures = tuple(
        float(getattr(state, 'temperature').value_in_unit(unit.kelvin))
        for state in thermodynamic_states
    )

    if output_mode == 'replica':
        output_paths = tuple(output_dir / f'output_{replica_index}.dcd' for replica_index in range(replica_thermodynamic_states.shape[1]))
        if not overwrite:
            _ensure_outputs_available(*output_paths)
        _write_replica_dcds(reporter, pdb, iterations, output_paths)
        _close_if_present(reporter)
        print('remd-extract mode: replica')
        for replica_index, path in enumerate(output_paths):
            print(f'replica {replica_index}: {path}')
        return output_paths
    if output_mode == 'temperature':
        output_paths = tuple(output_dir / f'output_T{temperature_index}.dcd' for temperature_index in range(len(temperatures)))
        manifest_path = output_dir / 'output_temperature_labels.txt'
        if not overwrite:
            _ensure_outputs_available(*output_paths, manifest_path)
        _write_temperature_dcds(reporter, pdb, iterations, replica_thermodynamic_states, temperatures, output_paths)
        _write_temperature_manifest(manifest_path, temperatures)
        _close_if_present(reporter)
        print('remd-extract mode: temperature')
        for temperature_index, (temperature, path) in enumerate(zip(temperatures, output_paths, strict=True)):
            print(f'T{temperature_index}: {temperature:.3f} K -> {path}')
        print(f'temperature map: {manifest_path}')
        return output_paths
    raise ValueError("output_mode must be either 'replica' or 'temperature'")


def build_remd_parser(subparsers, *, default_func) -> argparse.ArgumentParser:
    """Add the `cranberry remd` parser to a subparser collection."""

    parser = subparsers.add_parser(
        'remd',
        help='run CRANBERRY replica exchange MD',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('pdb', type=Path, help='canonical coarse-grained CRANBERRY PDB')
    parser.add_argument('--steps', type=int, required=True, default=argparse.SUPPRESS, help='total MD integration steps; REMD iterations are derived from steps/swap-steps')
    parser.add_argument('--output-dir', type=Path, default=Path('.'), help='directory for REMD outputs; defaults to current directory')
    parser.add_argument('--model', default='default', help='force-field model name; default resolves to cranberry-v1-alpha.1')
    parser.add_argument('--t-min', type=float, default=DEFAULT_REMD_T_MIN_K, help='minimum temperature in kelvin for parallel tempering')
    parser.add_argument('--t-max', type=float, default=DEFAULT_REMD_T_MAX_K, help='maximum temperature in kelvin for parallel tempering')
    parser.add_argument('--n-replicas', type=int, default=DEFAULT_REMD_N_REPLICAS, help='number of replicas in parallel tempering')
    parser.add_argument('--temperature-ladder', type=float, nargs='+', default=argparse.SUPPRESS, metavar='K', help='explicit replica temperatures in kelvin; overrides t_min/t_max/n_replicas')
    parser.add_argument('--swap-steps', type=int, default=DEFAULT_REMD_SWAP_STEPS, help='MD steps between replica-exchange attempts')
    parser.add_argument('--n-record', type=int, default=DEFAULT_REMD_N_RECORD, help='target number of NetCDF checkpoint records; checkpoint interval is derived in REMD iterations')
    parser.add_argument('--n-analysis', type=int, default=DEFAULT_REMD_N_ANALYSIS, help='target number of online-analysis records; 0 disables OpenMMTools online analysis')
    parser.add_argument('--extra-start-pdb', type=Path, default=None, help='additional canonical CG PDB whose coordinates seed alternating initial replicas')
    parser.add_argument('--salt', type=float, default=150.0, help='salt concentration in millimolar')
    parser.add_argument('--timestep', type=float, default=5.0, help='integration timestep in femtoseconds')
    parser.add_argument('--periodic', action='store_true', help='enable explicit periodic boundary conditions with a generated cubic box')
    parser.add_argument('--box-padding', type=float, default=2.0, help='periodic cubic box padding around the structure in nanometers')
    parser.add_argument('--platform', default='CPU', help="OpenMM platform name; use 'default' to let OpenMM choose")
    parser.add_argument('--restart-from', type=Path, default=None, help='OpenMMTools NetCDF storage to restart from')
    parser.add_argument('--overwrite', action='store_true', help='allow overwriting existing REMD NetCDF outputs; default is no-overwrite')
    _add_dcd_mode_options(parser)
    parser.add_argument('--write-dcd', action='store_true', help='translate the REMD NetCDF trajectory to DCD after the run')
    parser.set_defaults(func=default_func, workflow='remd')
    return parser


def _load_openmmtools():
    try:
        import openmmtools
        from openmmtools import mcmc, states
        import openmmtools.multistate as multistate
    except ImportError as exc:  # pragma: no cover - exercised in environments without the extra
        raise ImportError("REMD requires the optional 'remd' extra (openmmtools).") from exc
    return openmmtools, mcmc, states, multistate


def _load_mpiplus():
    try:
        import mpiplus
    except ImportError:
        return None
    return mpiplus


def _detect_mpi_runtime(mpiplus) -> _MpiRuntime:
    if mpiplus is None:
        return _MpiRuntime()
    try:
        comm = mpiplus.get_mpicomm()
    except Exception:
        return _MpiRuntime()
    if comm is None:
        return _MpiRuntime()

    rank_getter = getattr(comm, 'Get_rank', None)
    size_getter = getattr(comm, 'Get_size', None)
    rank = rank_getter() if callable(rank_getter) else getattr(comm, 'rank', 0)
    size = size_getter() if callable(size_getter) else getattr(comm, 'size', 1)
    return _MpiRuntime(enabled=int(size) > 1, rank=int(rank), size=int(size))


def _is_mpi_root(mpi_runtime: _MpiRuntime) -> bool:
    return not mpi_runtime.enabled or mpi_runtime.rank == 0


def _mpi_bcast(mpiplus, value, root: int = 0):
    if mpiplus is None:
        return value
    try:
        comm = mpiplus.get_mpicomm()
    except Exception:
        return value
    if comm is None:
        return value
    bcast = getattr(comm, 'bcast', None)
    if callable(bcast):
        return bcast(value, root=root)
    return value


def _sampler_from_storage(sampler_cls, storage_path: Path, mpiplus, mpi_runtime: _MpiRuntime):
    if mpiplus is None or not mpi_runtime.enabled:
        return sampler_cls.from_storage(str(storage_path))

    reporter = sampler_cls._reporter_from_storage(str(storage_path), check_exist=True)
    try:
        reporter.open(mode='r')
        sampler = sampler_cls._instantiate_sampler_from_reporter(reporter)
        sampler._restore_sampler_from_reporter(reporter)
    finally:
        reporter.close()

    _mpi_barrier(mpiplus)
    sampler._reporter = reporter
    mpiplus.run_single_node(0, sampler._reporter.open, mode='a', broadcast_result=False, sync_nodes=True)
    return sampler


def _mpi_barrier(mpiplus) -> None:
    comm = mpiplus.get_mpicomm()
    if comm is None:
        return
    barrier = getattr(comm, 'barrier', None)
    if callable(barrier):
        barrier()
        return
    barrier = getattr(comm, 'Barrier', None)
    if callable(barrier):
        barrier()


def _load_extra_start_pdb(pdb: app.PDBFile, extra_start_pdb: Path | None) -> app.PDBFile | None:
    if extra_start_pdb is None:
        return None
    extra_pdb = app.PDBFile(str(extra_start_pdb))
    if _topology_signature(extra_pdb.topology) != _topology_signature(pdb.topology):
        raise ValueError(
            'extra-start-pdb must have the same ordered atoms and bonds as the primary PDB'
        )
    return extra_pdb


def _topology_signature(topology: app.Topology):
    atoms = tuple(
        (
            atom.residue.chain.index,
            atom.residue.index,
            atom.residue.name,
            atom.name,
            atom.element.symbol if atom.element is not None else None,
        )
        for atom in topology.atoms()
    )
    bonds = tuple(
        sorted(
            (min(first.index, second.index), max(first.index, second.index))
            for first, second in topology.bonds()
        )
    )
    return atoms, bonds


def _validate_restart_metadata(
    args_path: Path,
    config: RemdRunConfig,
    model_name: str,
) -> None:
    if not args_path.exists():
        warnings.warn(
            f'REMD restart metadata not found at {args_path}; compatibility checks are limited to the stored temperature ladder.',
            RuntimeWarning,
            stacklevel=2,
        )
        return
    try:
        stored = json.loads(args_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Could not read REMD restart metadata from {args_path}: {exc}') from exc

    expected = {
        'pdb_sha256': _sha256(config.pdb_path),
        'model': model_name,
        'swap_steps': int(config.swap_steps),
        'salt_millimolar': float(config.salt_concentration_millimolar),
        'timestep_femtosecond': float(config.timestep_femtosecond),
        'periodic': bool(config.periodic),
        'box_padding_nanometer': float(config.box_padding_nanometer),
    }
    mismatches = []
    for key, value in expected.items():
        if key not in stored:
            mismatches.append(f'{key}=missing (requested {value!r})')
        elif stored[key] != value:
            mismatches.append(f'{key}={stored[key]!r} (requested {value!r})')
    if mismatches:
        raise ValueError(
            'REMD restart configuration does not match the existing args.json: '
            + '; '.join(mismatches)
        )


def _warn_platform_mismatch(requested_platform: str | None, actual_platform: str | None) -> None:
    if requested_platform is None or actual_platform is None:
        return
    if requested_platform != actual_platform:
        warnings.warn(
            f"Requested OpenMM platform {requested_platform!r}, but actual platform is {actual_platform!r}.",
            RuntimeWarning,
            stacklevel=2,
        )


def _apply_sampler_platform(sampler, openmmtools, platform: str | None) -> None:
    if platform is None:
        return
    platform_obj = mm.Platform.getPlatformByName(platform)
    sampler.energy_context_cache = openmmtools.cache.ContextCache(platform=platform_obj)


def _warn_cuda_online_analysis(actual_platform: str | None, online_analysis_interval: int | None) -> None:
    if online_analysis_interval is None or actual_platform != 'CUDA':
        return
    warnings.warn(
        'REMD online analysis uses PyMBAR/JAX, which may allocate CUDA GPU memory separately from OpenMM. '
        'If this run fails with a JAX CUDA out-of-memory error, use --n-analysis 0 or force JAX to CPU.',
        RuntimeWarning,
        stacklevel=2,
    )


def _actual_sampler_platform(sampler, reference_state) -> str | None:
    try:
        context, _integrator = sampler.energy_context_cache.get_context(reference_state)
        return context.getPlatform().getName()
    except Exception:
        return None


def _build_parallel_tempering_kwargs(ladder_spec: TemperatureLadderSpec, explicit_temperatures: tuple[float, ...] | None) -> dict[str, object]:
    if explicit_temperatures is not None:
        return {
            'temperatures': [temperature * unit.kelvin for temperature in explicit_temperatures],
            'n_temperatures': len(explicit_temperatures),
        }
    return {
        'min_temperature': ladder_spec.t_min * unit.kelvin,
        'max_temperature': ladder_spec.t_max * unit.kelvin,
        'n_temperatures': ladder_spec.n_replicas,
    }


def _build_remd_args(
    *,
    config: RemdRunConfig,
    model_name: str,
    output_netcdf_path: Path,
    output_dcd_path: Path | tuple[Path, ...] | None,
    output_dcd_manifest_path: Path | None,
    temperatures_kelvin: tuple[float, ...],
    n_iterations: int,
    checkpoint_interval: int,
    online_analysis_interval: int | None,
    actual_platform: str | None,
    openmmtools_version: str | None,
    mcmc_move: str,
    collision_rates: tuple[unit.Quantity, ...],
    pdb: app.PDBFile,
    system: mm.System,
) -> dict[str, object]:
    pdb_path = Path(config.pdb_path)
    ladder_spec = config.temperature_ladder
    explicit_temperatures = ladder_spec.resolve()
    box_vectors = pdb.topology.getPeriodicBoxVectors()
    return {
        'schema_version': 1,
        'run_kind': 'remd',
        'pdb_path': str(pdb_path),
        'pdb_sha256': _sha256(pdb_path),
        'extra_start_pdb_path': str(config.extra_start_pdb) if config.extra_start_pdb is not None else None,
        'extra_start_pdb_sha256': _sha256(config.extra_start_pdb) if config.extra_start_pdb is not None else None,
        'model': model_name,
        'steps': int(config.steps),
        'iterations': int(n_iterations),
        'swap_steps': int(config.swap_steps),
        'n_record': int(config.n_record),
        'checkpoint_interval': int(checkpoint_interval),
        'n_analysis': int(config.n_analysis),
        'online_analysis_interval': online_analysis_interval,
        'jax_platform_name_env': os.environ.get('JAX_PLATFORM_NAME'),
        'temperature_ladder_kelvin': [float(temperature) for temperature in temperatures_kelvin],
        'temperature_ladder_input_kelvin': list(explicit_temperatures) if explicit_temperatures is not None else None,
        't_min_kelvin': float(ladder_spec.t_min),
        't_max_kelvin': float(ladder_spec.t_max),
        'n_replicas': int(len(temperatures_kelvin)),
        'requested_n_replicas': int(ladder_spec.n_replicas),
        'salt_millimolar': float(config.salt_concentration_millimolar),
        'timestep_femtosecond': float(config.timestep_femtosecond),
        'langevin_collision_rate_per_ps': [
            float(collision_rate.value_in_unit(unit.picosecond**-1))
            for collision_rate in collision_rates
        ],
        'platform': config.platform,
        'actual_platform': actual_platform,
        'restart_from': str(config.restart_from) if config.restart_from is not None else None,
        'overwrite': bool(config.overwrite),
        'write_dcd': bool(config.write_dcd),
        'dcd_mode': config.dcd_mode,
        'periodic': bool(config.periodic),
        'box_padding_nanometer': float(config.box_padding_nanometer),
        'output_netcdf_path': str(output_netcdf_path),
        'output_dcd_path': _stringify_output_path(output_dcd_path),
        'output_dcd_manifest_path': str(output_dcd_manifest_path) if output_dcd_manifest_path is not None else None,
        'structure_summary': {
            'chains': sum(1 for _ in pdb.topology.chains()),
            'residues': pdb.topology.getNumResidues(),
            'atoms': pdb.topology.getNumAtoms(),
            'bonds': sum(1 for _ in pdb.topology.bonds()),
        },
        'periodic_box_vectors_present': bool(config.periodic and box_vectors is not None),
        'periodic_box_vectors_nanometer': (
            [
                [float(component.value_in_unit(unit.nanometer)) for component in vector]
                for vector in box_vectors
            ]
            if config.periodic and box_vectors is not None
            else None
        ),
        'force_groups': _present_force_group_names(system),
        'sampler': 'ParallelTemperingSampler',
        'mcmc_move': mcmc_move,
        'cranberry_version': __version__,
        'openmm_version': getattr(mm, '__version__', None),
        'openmmtools_version': openmmtools_version,
    }


def _stringify_output_path(path: Path | tuple[Path, ...] | None):
    if path is None:
        return None
    if isinstance(path, tuple):
        return [str(item) for item in path]
    return str(path)


def _expected_temperatures(ladder_spec: TemperatureLadderSpec) -> tuple[float, ...]:
    if ladder_spec.temperatures is not None:
        return ladder_spec.resolve() or ()
    if ladder_spec.n_replicas == 1:
        return (float(ladder_spec.t_min),)
    temperatures = np.logspace(
        np.log10(float(ladder_spec.t_min)),
        np.log10(float(ladder_spec.t_max)),
        num=ladder_spec.n_replicas,
    )
    return tuple(float(temperature) for temperature in temperatures)


def _add_dcd_mode_options(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--by-replica', dest='dcd_mode', action='store_const', const='replica', help='write one DCD per replica; this matches the legacy default extractor behavior')
    mode.add_argument('--by-temperature', dest='dcd_mode', action='store_const', const='temperature', help='write one DCD per thermodynamic temperature (output_T0, output_T1, ...)')
    parser.set_defaults(dcd_mode='replica')


def _expected_dcd_paths(output_dir: Path, output_mode: str, ladder_spec: TemperatureLadderSpec) -> tuple[Path, ...]:
    if output_mode == 'replica':
        resolved = ladder_spec.resolve()
        n_replicas = len(resolved) if resolved is not None else ladder_spec.n_replicas
        return tuple(output_dir / f'output_{replica_index}.dcd' for replica_index in range(n_replicas))
    if output_mode == 'temperature':
        resolved = ladder_spec.resolve()
        n_temperatures = len(resolved) if resolved is not None else ladder_spec.n_replicas
        return tuple(output_dir / f'output_T{temperature_index}.dcd' for temperature_index in range(n_temperatures))
    raise ValueError("output_mode must be either 'replica' or 'temperature'")


def _write_temperature_dcds(reporter, pdb: app.PDBFile, iterations, replica_thermodynamic_states, temperatures: tuple[float, ...], output_paths: tuple[Path, ...]) -> None:
    handles = []
    dcd_writers = []
    try:
        for path in output_paths:
            handle = path.open('wb')
            handles.append(handle)
            dcd_writers.append(app.DCDFile(handle, pdb.topology, 1 * unit.femtosecond))
        for temperature_index, _temperature in enumerate(temperatures):
            for frame_index, iteration in enumerate(iterations):
                state_samplers = reporter.read_sampler_states(iteration)
                replica_index = _find_replica_index(replica_thermodynamic_states, frame_index, temperature_index)
                sampler_state = state_samplers[replica_index]
                dcd_writers[temperature_index].writeModel(sampler_state.positions, periodicBoxVectors=getattr(sampler_state, 'box_vectors', None))
    finally:
        for handle in handles:
            handle.close()


def _write_temperature_manifest(manifest_path: Path, temperatures: tuple[float, ...]) -> None:
    lines = [
        '# REMD temperature labels',
        '# T{i} maps to the i-th thermodynamic state in the stored NetCDF ladder',
    ]
    for temperature_index, temperature in enumerate(temperatures):
        lines.append(f'T{temperature_index} = {temperature:.3f} K')
    manifest_path.write_text('\n'.join(lines) + '\n')


def _write_replica_dcds(reporter, pdb: app.PDBFile, iterations, output_paths: tuple[Path, ...]) -> None:
    handles = []
    dcd_writers = []
    try:
        for path in output_paths:
            handle = path.open('wb')
            handles.append(handle)
            dcd_writers.append(app.DCDFile(handle, pdb.topology, 1 * unit.femtosecond))
        for iteration in iterations:
            state_samplers = reporter.read_sampler_states(iteration)
            for replica_index, dcd in enumerate(dcd_writers):
                sampler_state = state_samplers[replica_index]
                dcd.writeModel(sampler_state.positions, periodicBoxVectors=getattr(sampler_state, 'box_vectors', None))
    finally:
        for handle in handles:
            handle.close()


def _ensure_outputs_available(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError('Refusing to overwrite existing REMD outputs: ' + ', '.join(existing))


def _read_temperatures_from_reporter(reporter) -> tuple[float, ...]:
    thermodynamic_states, _ = reporter.read_thermodynamic_states()
    temperatures: list[float] = []
    for state in thermodynamic_states:
        temperature = getattr(state, 'temperature', None)
        if temperature is None:
            raise ValueError('REMD storage does not contain temperature metadata')
        temperatures.append(float(temperature.value_in_unit(unit.kelvin)))
    return tuple(temperatures)


def _find_replica_index(replica_thermodynamic_states, iteration_index: int, temperature_index: int) -> int:
    replica_matches = np.where(replica_thermodynamic_states[iteration_index] == temperature_index)[0]
    if len(replica_matches) != 1:
        raise ValueError(
            f'expected exactly one replica for thermodynamic state {temperature_index} at iteration {iteration_index}'
        )
    return int(replica_matches[0])


def _close_if_present(obj) -> None:
    close = getattr(obj, 'close', None)
    if callable(close):
        close()
