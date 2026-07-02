# ADR 0002: Public API And CLI

## Status

Accepted.

## Context

The legacy `run_rna.py` exposes many research and tuning flags. The new package needs a small stable API that feels natural to OpenMM users while leaving room for advanced workflows later.

## Decision

Expose `CranberryForceField` as the public force-field object and follow OpenMM naming where appropriate, including `createSystem`.

Make OpenMM-native objects the primary API:

```python
pdb = app.PDBFile("input_cg_vs_conect.pdb")
ff = CranberryForceField("cranberry-v1")
system = ff.createSystem(pdb.topology, positions=pdb.positions, temperature=298)
```

Keep simulation setup separate from force-field construction. Provide high-level dataclass-backed helpers and CLI commands:

- `cranberry prepare`
- `cranberry cg`
- `cranberry md`
- `cranberry remd`
- `cranberry energy`
- `cranberry inspect`

Do not support old `core.*` imports in the new package. Provide migration docs instead of a compatibility API.

## Consequences

- The public API is smaller and easier to support.
- Existing legacy scripts need migration.
- CLI and Python behavior can stay aligned because CLI flags map to config dataclasses.
- Advanced workflows such as REMD, JAX, training, and PySAGES can be added without bloating the base install.
