from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from openmm import unit

try:
    __version__ = version("cranberry-rna")
except PackageNotFoundError:  # pragma: no cover - editable source tree before install
    __version__ = "0+unknown"
from ..data import available_forcefields
from ..energy import compute_energy
from ..forcefield import (
    FORCE_GROUP_NAMES,
    available_models,
    default_model_name,
    get_model_spec,
)
from ..md import run_md
from ..cg import coarse_grain_structure
from ..prepare import prepare_structure
from ..remd import RemdRunConfig, TemperatureLadderSpec, _add_dcd_mode_options, _detect_mpi_runtime, _is_mpi_root, _load_mpiplus, build_remd_parser, run_remd, translate_netcdf_to_dcd
from ..validation import validate_canonical_pdb

_NOOP_PREPARE_NOTE = (
    "Nothing to do: the current Phase 4 prepare workflow only changes the file when "
    "--add-terminal-phosphate is requested."
)

_TERMINAL_PHOSPHATE_NOTE = (
    "Many canonical CRANBERRY examples currently omit a 5'-terminal phosphate bead. "
    "Use --add-terminal-phosphate only if you want Cranberry to estimate sugar-puckering context near a chain end, "
    "because the coarse-grained pucker model uses phosphate context."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cranberry",
        description="CRANBERRY coarse-grained RNA simulation tools.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"cranberry-rna {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    _add_prepare_parser(subparsers, "prepare", "prepare a canonical CRANBERRY CG PDB")
    _add_cg_parser(subparsers, "cg", "coarse-grain an atomistic RNA PDB into canonical CRANBERRY CG form")
    build_remd_parser(subparsers, default_func=_remd)
    remd_extract_parser = subparsers.add_parser("remd-extract", help="translate REMD NetCDF output to DCD files", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    remd_extract_parser.add_argument("netcdf", type=Path, help="REMD NetCDF storage to translate")
    remd_extract_parser.add_argument("pdb", type=Path, help="canonical coarse-grained CRANBERRY PDB used to generate the NetCDF")
    remd_extract_parser.add_argument("--output-dir", type=Path, default=Path('.'), help="directory for extracted DCD files")
    _add_dcd_mode_options(remd_extract_parser)
    remd_extract_parser.add_argument("--overwrite", action="store_true", help="allow overwriting existing DCD files")
    remd_extract_parser.set_defaults(func=_remd_extract)

    md_parser = subparsers.add_parser("md", help="run CRANBERRY molecular dynamics", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    md_parser.add_argument("pdb", type=Path)
    md_parser.add_argument("--steps", type=int, required=True, default=argparse.SUPPRESS, help="number of MD integration steps; required")
    md_parser.add_argument("--output-dir", type=Path, default=Path("."), help="directory for MD outputs; defaults to current directory")
    md_parser.add_argument("--model", default="default", help="force-field model name; default resolves to cranberry-v1-alpha.1")
    md_parser.add_argument("--temperature", type=float, default=298.0, help="temperature in kelvin")
    md_parser.add_argument("--salt", type=float, default=150.0, help="salt concentration in millimolar")
    md_parser.add_argument("--timestep", type=float, default=5.0, help="integration timestep in femtoseconds")
    md_parser.add_argument("--periodic", action="store_true", help="enable explicit periodic boundary conditions with a generated cubic box")
    md_parser.add_argument("--box-padding", type=float, default=2.0, help="periodic cubic box padding around the structure in nanometers")
    md_parser.add_argument("--enforce-periodic-output", action="store_true", help="wrap MD DCD frames into the periodic box; force PBC remains controlled by --periodic")
    md_parser.add_argument("--no-log-progress", action="store_false", dest="log_progress", help="omit OpenMM Progress (%%) from the MD log file")
    md_parser.set_defaults(log_progress=True)
    md_parser.add_argument("--n-record", type=int, default=1000, help="target number of trajectory/log records; report interval is derived as max(1, steps // n_record)")
    md_parser.add_argument("--checkpoint-interval", type=int, default=None, help="MD checkpoint refresh interval in integration steps; defaults to 10 times the trajectory/log report interval")
    md_parser.add_argument("--write-minimization-report", action="store_true", help="write pre/post minimization energies to minimization_report.json")
    md_parser.add_argument("--platform", default="CPU", help="OpenMM platform name; use 'default' to let OpenMM choose")
    md_parser.add_argument("--restart-from", type=Path, default=None, help="OpenMM checkpoint to restart from")
    md_parser.add_argument("--no-overwrite", action="store_true", help="fail if default MD output files already exist")
    md_parser.set_defaults(func=_md)

    energy_parser = subparsers.add_parser("energy", help="compute total and decomposed CRANBERRY energies", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    energy_parser.add_argument("pdb", type=Path)
    energy_parser.add_argument("--model", default="default", help="force-field model name; default resolves to cranberry-v1-alpha.1")
    energy_parser.add_argument("--temperature", type=float, default=298.0, help="temperature in kelvin")
    energy_parser.add_argument("--salt", type=float, default=150.0, help="salt concentration in millimolar")
    energy_parser.add_argument("--platform", default="CPU", help="OpenMM platform name; use 'default' to let OpenMM choose")
    energy_parser.add_argument("--json", action="store_true", help="write energies as JSON")
    energy_parser.set_defaults(func=_energy)

    inspect_parser = subparsers.add_parser("inspect", help="inspect package data, models, or inputs", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    inspect_subparsers = inspect_parser.add_subparsers(dest="inspect_command", metavar="TARGET")
    inspect_parser.set_defaults(func=_inspect_summary)

    inspect_subparsers.add_parser("data", help="inspect packaged data").set_defaults(func=_inspect_data)

    ff_parser = inspect_subparsers.add_parser("forcefield", help="inspect force-field model assets")
    ff_parser.add_argument("model", nargs="?", default="default")
    ff_parser.set_defaults(func=_inspect_forcefield)

    input_parser = inspect_subparsers.add_parser("input", help="validate a canonical CRANBERRY input PDB")
    input_parser.add_argument("pdb", type=Path)
    input_parser.set_defaults(func=_inspect_input)

    return parser


def _add_prepare_parser(subparsers, command: str, help_text: str) -> None:
    parser = subparsers.add_parser(command, help=help_text, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pdb", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="prepared output PDB path; used only with --add-terminal-phosphate")
    parser.add_argument("--add-terminal-phosphate", action="store_true", help="insert a terminal phosphate at chain starts missing P")
    parser.add_argument("--no-overwrite", action="store_true", help="fail if the output file already exists")
    parser.set_defaults(func=_prepare, workflow=command)


def _add_cg_parser(subparsers, command: str, help_text: str) -> None:
    parser = subparsers.add_parser(command, help=help_text, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pdb", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="coarse-grained output PDB path; defaults to <input>_cg_vs_conect.pdb")
    parser.add_argument("--add-terminal-phosphate", action="store_true", help="insert a terminal phosphate at chain starts missing P after coarse-graining")
    parser.add_argument("--no-overwrite", action="store_true", help="fail if the output file already exists")
    parser.set_defaults(func=_cg, workflow=command)


def _prepare(args: argparse.Namespace) -> int:
    output_path = None
    if args.add_terminal_phosphate:
        output_path = args.output or args.pdb.with_name(f"{args.pdb.stem}_{args.workflow}.pdb")
    result = prepare_structure(
        args.pdb,
        output_path=output_path,
        add_terminal_phosphate=args.add_terminal_phosphate,
        overwrite=not args.no_overwrite,
    )
    print(f"input: {result.input_path}")
    print(f"valid: {result.output_validation.valid}")

    if not args.add_terminal_phosphate:
        print(_NOOP_PREPARE_NOTE)
        print(_TERMINAL_PHOSPHATE_NOTE)
        if args.output is not None:
            print(f"note: no output was written; ignored requested output path {args.output}")
        for warning in result.output_validation.warnings:
            print(f"warning: {warning}")
        for error in result.output_validation.errors:
            print(f"error: {error}")
        return 0 if result.output_validation.valid else 1

    print(f"output: {result.output_path}")
    print(f"added terminal phosphates: {result.inserted_terminal_phosphates}")
    for warning in result.output_validation.warnings:
        print(f"warning: {warning}")
    for error in result.output_validation.errors:
        print(f"error: {error}")
    return 0 if result.output_validation.valid else 1


def _cg(args: argparse.Namespace) -> int:
    output_path = args.output
    result = coarse_grain_structure(
        args.pdb,
        output_path=output_path,
        overwrite=not args.no_overwrite,
    )
    print(f"input: {result.input_path}")
    print(f"valid: {result.output_validation.valid}")
    print(f"output: {result.output_path}")
    print(f"coarse-grained residues: {result.residue_count}")
    print(f"virtual-site atoms: {result.inserted_virtual_sites}")
    if result.missing_terminal_phosphates:
        print(f"missing terminal phosphates: {result.missing_terminal_phosphates}")
    if args.add_terminal_phosphate:
        result_after_phosphate = prepare_structure(
            result.output_path,
            output_path=result.output_path,
            add_terminal_phosphate=True,
            overwrite=True,
        )
        print(f"added terminal phosphates: {result_after_phosphate.inserted_terminal_phosphates}")
        for warning in result_after_phosphate.output_validation.warnings:
            print(f"warning: {warning}")
        for error in result_after_phosphate.output_validation.errors:
            print(f"error: {error}")
        return 0 if result_after_phosphate.output_validation.valid else 1
    for warning in result.output_validation.warnings:
        print(f"warning: {warning}")
    for error in result.output_validation.errors:
        print(f"error: {error}")
    return 0 if result.output_validation.valid else 1


def _not_implemented(args: argparse.Namespace) -> int:
    raise SystemExit(
        f"The 'cranberry {args.command}' command is part of the v1 roadmap "
        "but is not implemented in this scaffold yet."
    )


def _remd(args: argparse.Namespace) -> int:
    platform = None if args.platform == "default" else args.platform
    temperatures = getattr(args, "temperature_ladder", None)
    ladder = TemperatureLadderSpec(
        temperatures=tuple(temperatures) if temperatures is not None else None,
        t_min=args.t_min,
        t_max=args.t_max,
        n_replicas=args.n_replicas,
    )
    result = run_remd(
        RemdRunConfig(
            pdb_path=args.pdb,
            steps=args.steps,
            output_dir=args.output_dir,
            model=args.model,
            temperature_ladder=ladder,
            swap_steps=args.swap_steps,
            n_record=args.n_record,
            n_analysis=args.n_analysis,
            salt_concentration_millimolar=args.salt,
            timestep_femtosecond=args.timestep,
            platform=platform,
            restart_from=args.restart_from,
            extra_start_pdb=args.extra_start_pdb,
            overwrite=args.overwrite,
            write_dcd=args.write_dcd,
            dcd_mode=args.dcd_mode,
            periodic=args.periodic,
            box_padding_nanometer=args.box_padding,
        )
    )
    if not _is_mpi_root(_detect_mpi_runtime(_load_mpiplus())):
        return 0
    print(
        "settings: "
        f"model={args.model}, "
        f"replicas={len(result.temperatures_kelvin)}, "
        f"steps={result.steps}, "
        f"iterations={result.iterations}, "
        f"swap_steps={result.swap_steps}, "
        f"checkpoint_interval={result.checkpoint_interval}, "
        f"platform={platform if platform is not None else 'default'}, "
        f"actual_platform={result.actual_platform if result.actual_platform is not None else 'unknown'}"
    )
    print(f"output directory: {result.output_dir}")
    print(f"netcdf: {result.output_netcdf_path}")
    print(f"args: {result.args_path}")
    if result.online_analysis_interval is not None:
        print(f"online analysis interval: {result.online_analysis_interval}")
        print(f"JAX_PLATFORM_NAME: {result.jax_platform_name_env if result.jax_platform_name_env is not None else 'unset'}")
    if result.output_dcd_path is not None:
        print(f"trajectory: {result.output_dcd_path}")
    if result.restart_from_path is not None:
        print(f"restarted from: {result.restart_from_path}")
    return 0


def _remd_extract(args: argparse.Namespace) -> int:
    output = translate_netcdf_to_dcd(
        args.netcdf,
        pdb_path=args.pdb,
        output_dir=args.output_dir,
        output_mode=args.dcd_mode,
        overwrite=args.overwrite,
    )
    if output is None:
        return 0
    print(f"output: {output}")
    return 0


def _runtime_settings_line(
args: argparse.Namespace, *, platform: str | None, report_interval: int | None = None, checkpoint_interval: int | None = None, actual_platform: str | None = None) -> str:
    model_spec = get_model_spec(args.model)
    effective_platform = platform if platform is not None else "default"
    return (
        "settings: "
        f"model={model_spec.name}, "
        f"temperature={args.temperature:.1f} K, "
        f"salt={args.salt:.1f} mM, "
        f"timestep={args.timestep:.1f} fs, "
        f"report_interval={report_interval if report_interval is not None else 'auto'}, "
        f"checkpoint_interval={checkpoint_interval if checkpoint_interval is not None else 'auto'}, "
        f"n_record={getattr(args, 'n_record', 'auto')}, "
        f"periodic={getattr(args, 'periodic', False)}, "
        f"enforce_periodic_output={getattr(args, 'enforce_periodic_output', False)}, "
        f"platform={effective_platform}, "
        f"actual_platform={actual_platform if actual_platform is not None else 'unknown'}"
    )


def _md(args: argparse.Namespace) -> int:
    platform = None if args.platform == "default" else args.platform
    result = run_md(
        args.pdb,
        steps=args.steps,
        output_dir=args.output_dir,
        model=args.model,
        temperature=args.temperature * unit.kelvin,
        salt_concentration=args.salt * unit.millimolar,
        timestep=args.timestep * unit.femtosecond,
        report_interval=getattr(args, "report_interval", None),
        checkpoint_interval=args.checkpoint_interval,
        n_record=args.n_record,
        platform=platform,
        restart_from=args.restart_from,
        overwrite=not args.no_overwrite,
        write_minimization_report=args.write_minimization_report,
        periodic=args.periodic,
        box_padding=args.box_padding * unit.nanometer,
        enforce_periodic_output=args.enforce_periodic_output,
        log_progress=args.log_progress,
    )
    print(_runtime_settings_line(args, platform=platform, report_interval=result.report_interval, checkpoint_interval=result.checkpoint_interval, actual_platform=result.actual_platform))
    print(f"output directory: {result.output_dir}")
    print(f"trajectory: {result.dcd_path}")
    print(f"log: {result.log_path}")
    print(f"detailed log: {result.detailed_log_path}")
    if result.minimization_report_path is not None:
        print(f"minimization report: {result.minimization_report_path}")
    if result.restart_from_path is not None:
        print(f"restarted from: {result.restart_from_path}")
    print(f"checkpoint: {result.checkpoint_path}")
    print(f"final pdb: {result.final_pdb_path}")
    return 0


def _energy(args: argparse.Namespace) -> int:
    platform = None if args.platform == "default" else args.platform
    report = compute_energy(
        args.pdb,
        model=args.model,
        temperature=args.temperature * unit.kelvin,
        salt_concentration=args.salt * unit.millimolar,
        platform=platform,
    )
    print(_runtime_settings_line(args, platform=platform, report_interval=getattr(args, "report_interval", None)))
    values = report.as_kj_per_mol()
    if args.json:
        print(json.dumps(values, indent=2, sort_keys=True))
    else:
        for name, value in values.items():
            print(f"{name:>14s}: {value: .8f} kJ/mol")
    return 0


def _inspect_summary(args: argparse.Namespace) -> int:
    print(f"cranberry-rna {__version__}")
    print(f"default model: {default_model_name()}")
    print("available models:")
    for model in available_models():
        print(f"  - {model}")
    print("inspect targets: data, forcefield, input")
    return 0


def _inspect_data(args: argparse.Namespace) -> int:
    print("packaged force-field files:")
    for filename in available_forcefields():
        print(f"  - {filename}")
    return 0


def _inspect_forcefield(args: argparse.Namespace) -> int:
    spec = get_model_spec(args.model)
    print(f"model: {spec.name}")
    print(f"description: {spec.description}")
    print(f"parameter file: {spec.parameter_path}")
    print(f"xml file: {spec.xml_path}")
    print("force groups:")
    for name in FORCE_GROUP_NAMES:
        print(f"  - {name}")
    return 0


def _inspect_input(args: argparse.Namespace) -> int:
    result = validate_canonical_pdb(args.pdb)
    print(f"input: {result.path}")
    print(f"valid: {result.valid}")
    print(f"atoms: {result.atom_count}")
    print(f"residues: {result.residue_count}")
    print(f"bonds: {result.bond_count}")
    print(f"frames: {result.frame_count}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    for error in result.errors:
        print(f"error: {error}")
    return 0 if result.valid else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    result = args.func(args)
    return int(result or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
