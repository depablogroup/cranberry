# ADR 0001: Package Layout

## Status

Accepted.

## Context

The legacy `OpenMM-CGRNA` project mixes runtime code, scripts, large datasets, notebooks, tests, profiling output, and research experiments. The new package should be installable as `cranberry-rna` while exposing `import cranberry`.

The workspace also keeps `OpenMM-CGRNA/` as a sibling reference folder during migration.

## Decision

Create a new project folder at `CRANBERRY/cranberry/`, parallel to `OpenMM-CGRNA/`.

Use a flat Python package layout:

```text
cranberry/
  pyproject.toml
  cranberry/
    __init__.py
    data/
  tests/
  docs/
  benchmarks/
```

The distribution name is `cranberry-rna`; the import package is `cranberry`.

Runtime data that must be available after pip/conda install lives inside the inner import package at `cranberry/cranberry/data/`.

## Consequences

- The new package can be developed without disturbing the legacy repo.
- pip/conda installation can include runtime assets reliably through package data.
- Developers must be careful about the two `cranberry/` levels: outer project folder versus inner import package.
- Public docs and developer docs need separate routing so design notes do not appear on the user-facing website by default.
