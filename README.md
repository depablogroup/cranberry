# cranberry-rna

`cranberry-rna` is the installable Python package for CRANBERRY coarse-grained RNA simulations with OpenMM. The distribution name is `cranberry-rna`; the Python import package is `cranberry`.

CRANBERRY is currently in alpha development while the stable v1 workflow is migrated from the legacy `OpenMM-CGRNA` project. The package now contains the canonical CRANBERRY v1 alpha force-field assets, an OpenMM-native force-field API, CPU energy decomposition, and a basic MD runner.

## Current Scope

Implemented:

- package import as `cranberry`
- packaged model assets: XML plus `cranberry-v1-alpha.1.h5`
- canonical input validation
- `CranberryForceField.createSystem()`
- `cranberry energy` and `cranberry.energy.compute_energy()`
- `cranberry md` and `cranberry.md.run_md()`
- default MD outputs: `output.dcd`, `log`, `detailed.log`, `args.json`, `checkpoint.chk`, `final.pdb`
- restart from OpenMM checkpoints

Planned:

- preparation/coarse-graining workflow
- optional 5'-phosphate insertion during preparation
- REMD through optional `openmmtools` support
- optional JAX/training/PySAGES workflows

## Development Install

Use the dedicated development environment, then install the package in editable mode:

```bash
conda activate cranberry-dev
pip install -e ".[dev]"
```

Confirm the import points at the editable checkout:

```bash
python -c 'import cranberry; print(cranberry.__file__)'
```

## Quickstart

Inspect the package and default model:

```bash
cranberry inspect
cranberry inspect forcefield
cranberry inspect data
```

Validate a canonical coarse-grained input PDB:

```bash
cranberry inspect input cranberry/data/examples/2ntCG_cg_vs_conect.pdb
```

Compute total and decomposed energies on CPU:

```bash
cranberry energy cranberry/data/examples/2ntCG_cg_vs_conect.pdb --platform CPU
```

Run a short CPU MD simulation:

```bash
cranberry md cranberry/data/examples/2ntCG_cg_vs_conect.pdb \
  --steps 1000 \
  --report-interval 100 \
  --output-dir md-out \
  --platform CPU
```

Restart from the generated checkpoint. Restart appends to `output.dcd`, `log`, and `detailed.log` when they exist; if any are missing, Cranberry warns and creates them starting from the checkpoint step. `args.json` records latest run metadata, and distinct previous metadata is archived under `args_history/`.

```bash
cranberry md cranberry/data/examples/2ntCG_cg_vs_conect.pdb \
  --steps 1000 \
  --restart-from md-out/checkpoint.chk \
  --output-dir md-out \
  --platform CPU
```

## Python API

The API follows OpenMM conventions and accepts OpenMM unit quantities where appropriate.

```python
from openmm import app, unit
from cranberry import CranberryForceField

pdb = app.PDBFile("input_cg_vs_conect.pdb")
forcefield = CranberryForceField()
system = forcefield.createSystem(
    pdb.topology,
    positions=pdb.positions,
    temperature=298 * unit.kelvin,
    salt_concentration=150 * unit.millimolar,
)
```

For the high-level MD runner:

```python
from cranberry.md import run_md

result = run_md(
    "input_cg_vs_conect.pdb",
    steps=1000,
    output_dir="md-out",
    platform="CPU",
)
print(result.checkpoint_path)
```

## Tests And Docs

Run the default CPU test suite:

```bash
python -m pytest -q
```

Build the Sphinx docs:

```bash
sphinx-build -b html docs docs/_build/html
```

Build package artifacts:

```bash
python -m build
```

## Documentation

Primary documentation lives under `docs/`. Developer-only design notes and code-review reports live under `docs/dev/` and are intentionally excluded from the public documentation narrative.

Useful entry points:

- `docs/quickstart.md`
- `docs/reference/cli.md`
- `docs/reference/api.md`
- `docs/reference/outputs.md`
- `docs/dev/cranberry-v1-plan.md`

## License

MIT.
