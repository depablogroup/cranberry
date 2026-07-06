# Energy Decomposition

Use `cranberry energy` to compute total and force-group potential energies for a canonical CRANBERRY coarse-grained PDB.

The input must be a prepared CG PDB with `BC`/`BN` virtual-site atoms and `CONECT` records.

## Command Line

```bash
cranberry energy input_cg_vs_conect.pdb --platform CPU
```

For machine-readable output:

```bash
cranberry energy input_cg_vs_conect.pdb --platform CPU --json
```

Common options:

```text
--model MODEL          force-field model; default resolves to cranberry-v1-alpha.1
--temperature K        temperature in kelvin, default 298
--salt MM              salt concentration in millimolar, default 150
--platform NAME        OpenMM platform name, default CPU; use default to let OpenMM choose
--json                 print JSON instead of a text table
```

## Python API

```python
from cranberry.energy import compute_energy

report = compute_energy(
    "input_cg_vs_conect.pdb",
    temperature=298,
    salt_concentration=150,
    platform="CPU",
)

energies = report.as_kj_per_mol()
print(energies["total"])
print(energies["bond"])
```

The returned dictionary uses `kJ/mol` values and includes named components such as:

```text
total
bond
angle
dihedral
pucker
stacking35
stacking55
stacking33
pairing
wca
spline
electrostatic
```

## Current Reference Semantics

The Phase 2 energy regression tests are matched against the GPU workshop copy of `OpenMM-CGRNA` using:

```bash
run_rna.py --dry-run --post-run-reporting --use-pdb
```

That legacy post-run path evaluates the coordinates already present in the input PDB. For parity, `compute_energy()` also uses the PDB-provided `BC`/`BN` virtual-site coordinates rather than recomputing virtual sites inside the OpenMM context.

In legacy output, sugar energy is split into `pucker` and `terminal_U3`. Cranberry reports their sum as the `pucker` component.
