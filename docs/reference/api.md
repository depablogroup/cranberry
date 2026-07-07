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
    timestep=10 * unit.femtosecond,
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

`run_md` and `create_simulation` pass one temperature to both force-field construction and the Langevin integrator, so electrostatics and dynamics stay consistent.
