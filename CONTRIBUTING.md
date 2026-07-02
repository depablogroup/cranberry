# Contributing

## Development Environment

Use the dedicated development environment for the new package:

```bash
conda activate cranberry-dev
python -m pip install -e ".[dev]"
```

If the environment does not exist yet, create it with Python 3.11:

```bash
conda create -n cranberry-dev python=3.11
conda activate cranberry-dev
python -m pip install -e ".[dev]"
```

## Verify Changes

Run these from the repository root:

```bash
python -m pytest -q
sphinx-build -b html docs docs/_build/html
cranberry --help
```

## Repository Layout

- `cranberry/`: import package for `cranberry-rna`.
- `tests/`: package tests.
- `docs/`: public documentation source.
- `docs/dev/`: developer-only design notes and ADRs. These are kept in GitHub but excluded from the public Sphinx build.
- `benchmarks/`: benchmark scripts and result snapshots as they are added.

## Project Context

The legacy `OpenMM-CGRNA/` repository lives beside this project in the parent workspace and should be treated as reference material by default. Temporary reference outputs should be generated from the GPU workshop copy documented in `docs/dev/reference-output-generation.md`.

Before changing package architecture, force-field behavior, CLI contracts, output formats, or tests, read `docs/dev/cranberry-v1-plan.md`.
