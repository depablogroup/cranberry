from __future__ import annotations

import argparse
import json
from pathlib import Path

from cranberry import __version__
from cranberry.data import available_forcefields
from openmm import unit

from cranberry.energy import compute_energy
from cranberry.md import run_md
from cranberry.forcefield import (
    FORCE_GROUP_NAMES,
    available_models,
    default_model_name,
    get_model_spec,
)
from cranberry.validation import validate_canonical_pdb

_COMMANDS = ("prepare", "cg", "remd")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cranberry",
        description="CRANBERRY coarse-grained RNA simulation tools.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"cranberry-rna {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for command in _COMMANDS:
        subparser = subparsers.add_parser(
            command,
            help=f"{command} workflow (not implemented yet)",
        )
        subparser.set_defaults(func=_not_implemented)

    md_parser = subparsers.add_parser("md", help="run CRANBERRY molecular dynamics")
    md_parser.add_argument("pdb", type=Path)
    md_parser.add_argument("--steps", type=int, required=True, help="number of MD integration steps")
    md_parser.add_argument("--output-dir", type=Path, default=Path("."), help="directory for MD outputs; defaults to current directory")
    md_parser.add_argument("--model", default="default")
    md_parser.add_argument("--temperature", type=float, default=298.0, help="temperature in kelvin")
    md_parser.add_argument("--salt", type=float, default=150.0, help="salt concentration in millimolar")
    md_parser.add_argument("--timestep", type=float, default=10.0, help="integration timestep in femtoseconds")
    md_parser.add_argument("--report-interval", type=int, default=None, help="steps between output reports; defaults to min(steps, 1000)")
    md_parser.add_argument("--platform", default="CPU", help="OpenMM platform name; use 'default' to let OpenMM choose")
    md_parser.add_argument("--restart-from", type=Path, default=None, help="OpenMM checkpoint to restart from")
    md_parser.add_argument("--no-overwrite", action="store_true", help="fail if default MD output files already exist")
    md_parser.set_defaults(func=_md)

    energy_parser = subparsers.add_parser("energy", help="compute total and decomposed CRANBERRY energies")
    energy_parser.add_argument("pdb", type=Path)
    energy_parser.add_argument("--model", default="default")
    energy_parser.add_argument("--temperature", type=float, default=298.0, help="temperature in kelvin")
    energy_parser.add_argument("--salt", type=float, default=150.0, help="salt concentration in millimolar")
    energy_parser.add_argument("--platform", default="CPU", help="OpenMM platform name; use 'default' to let OpenMM choose")
    energy_parser.add_argument("--json", action="store_true", help="write energies as JSON")
    energy_parser.set_defaults(func=_energy)

    inspect_parser = subparsers.add_parser("inspect", help="inspect package data, models, or inputs")
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


def _not_implemented(args: argparse.Namespace) -> int:
    raise SystemExit(
        f"The 'cranberry {args.command}' command is part of the v1 roadmap "
        "but is not implemented in this scaffold yet."
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
        report_interval=args.report_interval,
        platform=platform,
        restart_from=args.restart_from,
        overwrite=not args.no_overwrite,
    )
    print(f"output directory: {result.output_dir}")
    print(f"trajectory: {result.dcd_path}")
    print(f"log: {result.log_path}")
    print(f"detailed log: {result.detailed_log_path}")
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
