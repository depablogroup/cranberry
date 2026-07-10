# API Reference

The public API is OpenMM-native. The main force-field entry point is `CranberryForceField`:

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

`CranberryForceField.createSystem()` constructs an OpenMM `System`.

For coarse-graining atomistic RNA input into canonical CRANBERRY form:

```python
from cranberry.cg import coarse_grain_structure

result = coarse_grain_structure(
    "atomistic_input.pdb",
    output_path="atomistic_input_cg_vs_conect.pdb",
)
print(result.output_path)
```

For preparation and normalization of an already coarse-grained input:

```python
from cranberry.prepare import prepare_structure

result = prepare_structure(
    "input_cg_vs_conect.pdb",
    add_terminal_phosphate=True,
    output_path="prepared.pdb",
)
print(result.output_path)
```

For one-shot energy evaluation:

```python
from cranberry.energy import compute_energy

report = compute_energy("input_cg_vs_conect.pdb", platform="CPU")
print(report.as_kj_per_mol())
```

The energy report contains total potential energy and named force-group components.

For MD, use `run_md` when you want Cranberry's default reporters and output names:

```python
from openmm import unit
from cranberry.md import run_md

result = run_md(
    "input_cg_vs_conect.pdb",
    steps=1000,
    output_dir="md-out",
    temperature=298 * unit.kelvin,
    timestep=5 * unit.femtosecond,
    platform="CPU",
)
print(result.final_pdb_path)
```

Restart from a checkpoint by passing `restart_from`. Restart appends to `output.dcd`, `log`, and `detailed.log` in the output directory; missing output files are created with a warning:

```python
restart = run_md(
    "input_cg_vs_conect.pdb",
    steps=1000,
    output_dir="md-out",
    restart_from="md-out/checkpoint.chk",
    platform="CPU",
)
```

Use `create_simulation` when you need direct access to an OpenMM `Simulation` before running dynamics:

```python
from cranberry.md import create_simulation

simulation = create_simulation("input_cg_vs_conect.pdb", platform="CPU")
simulation.step(1000)
```

`prepare_structure` currently targets already coarse-grained CRANBERRY PDBs. It is the API behind `cranberry prepare`, with `cranberry cg` as the upstream coarse-graining command. Without terminal-phosphate insertion it performs validation only and returns without writing a new file. With terminal-phosphate insertion it rewrites the PDB, adds a `P` bead at each chain start that is missing one, adds the corresponding `P-S3` bond, and writes canonical `CONECT` records. This option restores phosphate context near a chain end when that context matters to the coarse-grained sugar-puckering estimate, and the placement heuristic is treated as fixed v1 behavior.

`run_md` and `create_simulation` pass one temperature to both force-field construction and the Langevin integrator, so electrostatics and dynamics stay consistent.

## REMD API

The REMD sketch below is generated from the module docstrings and public signatures.

```{automodule} cranberry.remd
:members:
:undoc-members:
:show-inheritance:
```
