# API Reference

The public API is OpenMM-native. The main entry point is `CranberryForceField`:

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

`CranberryForceField.createSystem()` constructs an OpenMM `System`. It does not run dynamics.

For one-shot energy evaluation:

```python
from cranberry.energy import compute_energy

report = compute_energy("input_cg_vs_conect.pdb", platform="CPU")
print(report.as_kj_per_mol())
```

The energy report contains total potential energy and named force-group components.
