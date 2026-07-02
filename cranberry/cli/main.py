from __future__ import annotations

import argparse
from pathlib import Path

from cranberry import __version__
from cranberry.data import available_forcefields
from cranberry.forcefield import (
    FORCE_GROUP_NAMES,
    available_models,
    default_model_name,
    get_model_spec,
)
from cranberry.validation import validate_canonical_pdb

_COMMANDS = ("prepare", "cg", "md", "remd", "energy")


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
