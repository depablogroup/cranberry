from __future__ import annotations

import argparse

from cranberry import __version__


_COMMANDS = ("prepare", "cg", "md", "remd", "energy", "inspect")


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

    return parser


def _not_implemented(args: argparse.Namespace) -> int:
    raise SystemExit(
        f"The 'cranberry {args.command}' command is part of the v1 roadmap "
        "but is not implemented in this scaffold yet."
    )


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
