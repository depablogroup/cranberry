# Force Field

The v1 force field is the canonical CRANBERRY model. 5SPN and RACER are not public v1 APIs.

Current packaged model:

```text
cranberry-v1-alpha.1
```

The package resolves `default` to the current alpha model during development.

The alpha model is stored as one packaged HDF5 file:

```text
cranberry/data/forcefields/cranberry-v1-alpha.1.h5
```

This file contains the canonical bonded, sugar-pucker, WCA, spline, stacking, and pairing parameters needed by `CranberryForceField`. The legacy `run_rna.py --angle-scaling 0.1` behavior is baked into the packaged angle spring constants rather than exposed as a public option.

Named energy components are:

```text
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

Force-group integer IDs are internal and should not be treated as public API.
